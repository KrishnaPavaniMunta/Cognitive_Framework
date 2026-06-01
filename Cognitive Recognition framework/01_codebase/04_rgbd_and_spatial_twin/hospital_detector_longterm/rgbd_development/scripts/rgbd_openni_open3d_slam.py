from __future__ import annotations

import argparse
import base64
import json
import queue
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from flask import Flask, Response, render_template_string, request
from openni import openni2


WIDTH = 640
HEIGHT = 480
FPS = 30

# ── Shared frame queues (maxsize=2 keeps delivery fresh) ──────────────────────
_rgb_q:   queue.Queue = queue.Queue(maxsize=2)
_depth_q: queue.Queue = queue.Queue(maxsize=2)
_stop_evt: threading.Event = threading.Event()
_stats: dict = {"frame": 0, "x": 0.0, "y": 0.0, "z": 0.0, "pts": 0}

# Point cloud payload for Three.js (updated every update_every frames)
_pcd_lock = threading.Lock()
_pcd_build_lock = threading.Lock()
_pcd_payload: dict = {"n": 0, "v": 0, "pts_b64": "", "col_b64": ""}

# ── HTML dashboard ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>HospitalGuard SLAM</title>
  <script type="importmap">
  {"imports":{
    "three":"https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.min.js",
    "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/"
  }}
  </script>
  <style>
    :root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--accent:#58a6ff;
          --text:#c9d1d9;--muted:#8b949e;--red:#f85149;}
    *{box-sizing:border-box;margin:0;padding:0;}
    html,body{width:100%;height:100%;overflow:hidden;background:var(--bg);
      color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
      display:flex;flex-direction:column;}

    /* ── header ── */
    #hdr{height:42px;padding:0 16px;background:var(--panel);
      border-bottom:1px solid var(--border);
      display:flex;align-items:center;gap:14px;flex-shrink:0;}
    #hdr h1{font-size:.95rem;font-weight:700;color:var(--accent);white-space:nowrap;}
    #stats{display:flex;gap:16px;font-size:.75rem;color:var(--muted);flex:1;}
    #stats b{color:var(--text);}
    #stop-btn{padding:4px 12px;background:var(--red);color:#fff;border:none;
      border-radius:5px;cursor:pointer;font-size:.8rem;font-weight:600;
      transition:opacity .15s;flex-shrink:0;}
    #stop-btn:hover{opacity:.75;} #stop-btn:disabled{opacity:.4;cursor:default;}

    /* ── main grid: RGB top row, bottom row below ── */
    #main{flex:1;min-height:0;display:grid;
      grid-template-rows:38% 1fr;gap:8px;padding:8px;}

    /* ── panels ── */
    .panel{background:var(--panel);border:1px solid var(--border);
      border-radius:8px;overflow:hidden;display:flex;flex-direction:column;}
    .plabel{padding:3px 10px;font-size:.65rem;font-weight:700;color:var(--muted);
      text-transform:uppercase;letter-spacing:.07em;
      background:rgba(255,255,255,.025);border-bottom:1px solid var(--border);
      flex-shrink:0;}
    canvas.feed{width:100%;height:100%;display:block;background:#000;}

    /* ── bottom row: depth left, 3D map right (large) ── */
    #bottom{display:grid;grid-template-columns:30% 1fr;gap:8px;min-height:0;}

    /* ── Three.js map wrapper ── */
    #map-wrap{position:relative;background:#111318;border:1px solid var(--border);
      border-radius:8px;overflow:hidden;display:flex;flex-direction:column;}
    #map-canvas{display:block;width:100%;flex:1;min-height:0;}
    #map-hint{position:absolute;bottom:7px;right:10px;font-size:.62rem;
      color:rgba(255,255,255,.28);pointer-events:none;line-height:1.6;text-align:right;}

    /* ── status bar ── */
    #sb{height:22px;padding:0 16px;font-size:.68rem;color:var(--muted);
      background:var(--panel);border-top:1px solid var(--border);
      display:flex;align-items:center;flex-shrink:0;}
    #sb span{color:var(--accent);}
  </style>
</head>
<body>
  <div id="hdr">
    <h1>&#9679; HospitalGuard SLAM</h1>
    <div id="stats">
      <span>Frame <b id="f">—</b></span>
      <span>X <b id="cx">—</b></span>
      <span>Y <b id="cy">—</b></span>
      <span>Z <b id="cz">—</b></span>
      <span>Map pts <b id="pts">—</b></span>
    </div>
    <button id="stop-btn" onclick="stopSlam()">Stop &amp; Save</button>
  </div>

  <div id="main">
    <!-- Row 1: RGB full width -->
    <div class="panel">
      <div class="plabel">RGB — Live</div>
      <canvas id="rgb-canvas" class="feed"></canvas>
    </div>

    <!-- Row 2: Depth + 3D Map -->
    <div id="bottom">
      <div class="panel">
        <div class="plabel">Depth — Turbo</div>
        <canvas id="depth-canvas" class="feed"></canvas>
      </div>
      <div id="map-wrap">
        <div class="plabel" style="flex-shrink:0">3D Map — WebGL GPU</div>
        <canvas id="map-canvas"></canvas>
        <div id="map-hint">
          Left drag &nbsp;orbit<br>
          Right drag &nbsp;pan<br>
          Scroll &nbsp;zoom<br>
          Double-click &nbsp;focus
        </div>
      </div>
    </div>
  </div>

  <div id="sb">Waiting for first frame&hellip;</div>

  <!-- Hidden MJPEG sources (decoded by browser hardware pipeline) -->
  <img id="rgb-src"   src="/stream/rgb"   style="display:none"/>
  <img id="depth-src" src="/stream/depth" style="display:none"/>

  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    // ─────────────────────────────────────────────────────────────────────────
    // GPU canvas rendering for RGB + Depth feeds
    // Hidden <img> elements carry the MJPEG streams; we blit to canvas every RAF
    // ─────────────────────────────────────────────────────────────────────────
    const rgbCv  = document.getElementById('rgb-canvas');
    const depCv  = document.getElementById('depth-canvas');
    const rgbCtx = rgbCv.getContext('2d', { alpha: false });
    const depCtx = depCv.getContext('2d', { alpha: false });
    const rgbSrc = document.getElementById('rgb-src');
    const depSrc = document.getElementById('depth-src');

    function syncSize(cv) {
      const r = cv.getBoundingClientRect();
      const w = Math.max(1, Math.floor(r.width));
      const h = Math.max(1, Math.floor(r.height));
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    }
    function drawContain(ctx, img, cw, ch) {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, cw, ch);
      if (!img || img.naturalWidth <= 0 || img.naturalHeight <= 0) return;
      const s = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
      const dw = Math.floor(img.naturalWidth * s);
      const dh = Math.floor(img.naturalHeight * s);
      const dx = Math.floor((cw - dw) * 0.5);
      const dy = Math.floor((ch - dh) * 0.5);
      ctx.drawImage(img, dx, dy, dw, dh);
    }
    function drawFeeds() {
      syncSize(rgbCv);
      syncSize(depCv);
      drawContain(rgbCtx, rgbSrc, rgbCv.width, rgbCv.height);
      drawContain(depCtx, depSrc, depCv.width, depCv.height);
      requestAnimationFrame(drawFeeds);
    }
    requestAnimationFrame(drawFeeds);

    // ─────────────────────────────────────────────────────────────────────────
    // Three.js WebGL point-cloud viewer
    // ─────────────────────────────────────────────────────────────────────────
    const mapCanvas = document.getElementById('map-canvas');
    const renderer = new THREE.WebGLRenderer({
      canvas: mapCanvas, antialias: true, powerPreference: 'high-performance'
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x111318, 1);

    const scene = new THREE.Scene();

    // Floor grid (Blender-style)
    const grid = new THREE.GridHelper(20, 40, 0x2a2a3c, 0x1e1e2e);
    scene.add(grid);

    // World axes (X=red, Y=green, Z=blue)
    scene.add(new THREE.AxesHelper(0.5));

    // Perspective camera
    const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 200);
    camera.position.set(0, 3, 5);
    camera.lookAt(0, 0, 0);

    // OrbitControls with smooth damping (Blender-like feel)
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping  = true;
    controls.dampingFactor  = 0.06;
    controls.screenSpacePanning = true;
    controls.minDistance    = 0.05;
    controls.maxDistance    = 80;
    controls.zoomSpeed      = 1.2;
    controls.rotateSpeed    = 0.8;
    // Double-click to focus on centroid
    renderer.domElement.addEventListener('dblclick', () => {
      if (geo.boundingSphere) {
        controls.target.copy(geo.boundingSphere.center);
        controls.update();
      }
    });

    // Point cloud geometry + material
    const geo = new THREE.BufferGeometry();
    const mat = new THREE.PointsMaterial({
      size: 0.018, vertexColors: true, sizeAttenuation: true,
    });
    const pcdPoints = new THREE.Points(geo, mat);
    scene.add(pcdPoints);

    // Resize map renderer to fill its container
    function resizeMap() {
      const wrap = mapCanvas.parentElement;
      const label = wrap.querySelector('.plabel');
      const ww = Math.max(1, wrap.clientWidth);
      const wh = Math.max(1, wrap.clientHeight - label.offsetHeight);
      renderer.setSize(ww, wh);
      camera.aspect = ww / wh;
      camera.updateProjectionMatrix();
    }
    resizeMap();
    window.addEventListener('resize', resizeMap);

    // GPU render loop — runs at 60fps independent of data updates
    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // ─────────────────────────────────────────────────────────────────────────
    // Point-cloud polling: fetch compact binary JSON from /api/pcd
    // Server only sends new data when the point cloud has changed (version gate)
    // ─────────────────────────────────────────────────────────────────────────
    let lastV = -1;
    async function fetchPcd() {
      try {
        const res  = await fetch('/api/pcd?v=' + lastV);
        const data = await res.json();
        if (data.same || data.n === 0) return;
        lastV = data.v;

        // Float32 XYZ positions
        const ptsRaw = Uint8Array.from(atob(data.pts_b64), c => c.charCodeAt(0));
        const positions = new Float32Array(ptsRaw.buffer);

        // Uint8 RGB → Float32 [0,1]
        const colRaw = Uint8Array.from(atob(data.col_b64), c => c.charCodeAt(0));
        const colors = new Float32Array(data.n * 3);
        for (let i = 0; i < colRaw.length; i++) colors[i] = colRaw[i] / 255;

        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
        geo.computeBoundingSphere();
      } catch (_) { /* network hiccup, skip */ }
    }
    setInterval(fetchPcd, 250);

    // ─────────────────────────────────────────────────────────────────────────
    // SSE stats bar
    // ─────────────────────────────────────────────────────────────────────────
    const sse = new EventSource('/stats');
    sse.onmessage = e => {
      const d = JSON.parse(e.data);
      document.getElementById('f').textContent   = d.frame;
      document.getElementById('cx').textContent  = d.x.toFixed(3);
      document.getElementById('cy').textContent  = d.y.toFixed(3);
      document.getElementById('cz').textContent  = d.z.toFixed(3);
      document.getElementById('pts').textContent = d.pts.toLocaleString();
      document.getElementById('sb').innerHTML =
        `Frame <span>${d.frame}</span> &nbsp;|&nbsp; ` +
        `Cam (${d.x.toFixed(3)}, ${d.y.toFixed(3)}, ${d.z.toFixed(3)}) m &nbsp;|&nbsp; ` +
        `${d.pts.toLocaleString()} map points`;
    };

    window.stopSlam = () => {
      const b = document.getElementById('stop-btn');
      b.textContent = 'Saving\u2026'; b.disabled = true;
      fetch('/stop');
    };
  </script>
</body>
</html>"""

# ── Flask app ─────────────────────────────────────────────────────────────────
_app = Flask(__name__)


def _push(q: queue.Queue, img: np.ndarray, quality: int = 82) -> None:
    """JPEG-encode img and push to queue, dropping the oldest frame if full."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return
    data = buf.tobytes()
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(data)
    except queue.Full:
        pass


def _mjpeg(q: queue.Queue):
    """Generator that yields MJPEG boundary chunks from a frame queue."""
    while not _stop_evt.is_set():
        try:
            data = q.get(timeout=0.5)
        except queue.Empty:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"


@_app.route("/")
def _index():
    return render_template_string(_HTML)


@_app.route("/stream/rgb")
def _stream_rgb():
    return Response(_mjpeg(_rgb_q), mimetype="multipart/x-mixed-replace; boundary=frame")


@_app.route("/stream/depth")
def _stream_depth():
    return Response(_mjpeg(_depth_q), mimetype="multipart/x-mixed-replace; boundary=frame")


@_app.route("/api/pcd")
def _api_pcd():
    """Return latest point cloud as compact base64-encoded binary JSON.
    Uses a version gate so the browser only downloads when the PCD changes."""
    client_v = request.args.get("v", -1, type=int)
    with _pcd_lock:
        payload = _pcd_payload.copy()
    if client_v == payload["v"] and client_v >= 0:
        return Response('{"same":true}', mimetype="application/json")
    return Response(json.dumps(payload), mimetype="application/json")


@_app.route("/stats")
def _stats_sse():
    def _gen():
        last = -1
        while not _stop_evt.is_set():
            if _stats["frame"] != last:
                last = _stats["frame"]
                yield f"data: {json.dumps(_stats)}\n\n"
            time.sleep(0.08)
    return Response(
        _gen(), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@_app.route("/stop")
def _stop_route():
    _stop_evt.set()
    return "ok"


# ── Camera / RGBD helpers ──────────────────────────────────────────────────────

def _open_depth_stream(device: openni2.Device):
    """Start OpenNI depth stream (Astra colour must go through cv2 instead)."""
    if device.get_sensor_info(openni2.SENSOR_DEPTH) is None:
        raise RuntimeError("Depth sensor is not available on this OpenNI device.")
    ds = device.create_depth_stream()
    ds.configure_mode(WIDTH, HEIGHT, FPS, openni2.PIXEL_FORMAT_DEPTH_1_MM)
    ds.start()
    time.sleep(0.5)   # let USB settle before cv2 opens colour camera
    return ds


def _open_color_camera(color_index: int = -1) -> cv2.VideoCapture:
    """Open colour camera. When color_index=-1, tries index 1 only (Astra UVC).
    Indices 2-4 can crash Python via a C++ fault when OpenNI holds the USB bus,
    so we never scan beyond index 1 unless the user explicitly specifies another."""
    # When auto-detecting, only try index 1 — the Astra UVC colour camera.
    # Scanning 2-4 with DSHOW while OpenNI owns the bus causes a silent C++ crash.
    indices  = [1] if color_index < 0 else [color_index]
    backends = (cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF)

    for attempt in range(4):          # up to 4 attempts with backoff
        for idx in indices:
            for backend in backends:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                ok, frame = cap.read()
                if ok and frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                    print(f"[SLAM] Colour camera opened at index={idx} backend={backend}", flush=True)
                    return cap
                cap.release()
        if attempt < 3:
            print(f"[SLAM] Colour camera not ready, retrying in 2 s… (attempt {attempt+1}/4)", flush=True)
            time.sleep(2.0)

    idx_str = "1 (auto)" if color_index < 0 else str(color_index)
    raise RuntimeError(
        f"Failed to open colour camera at index {idx_str}. "
        "Camera may still be releasing from a previous session. "
        "Wait a few seconds and retry, or use --color-index 1."
    )


def _depth_colormap(depth_frame) -> np.ndarray:
    """OpenNI depth frame → BGR Turbo colourmap image (for display)."""
    raw = np.frombuffer(
        depth_frame.get_buffer_as_uint16(), dtype=np.uint16, count=WIDTH * HEIGHT
    ).reshape((HEIGHT, WIDTH)).copy()
    norm = np.clip(raw.astype(np.float32) / 5000.0 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _pcd_to_payload(pcd: o3d.geometry.PointCloud, version: int) -> dict:
    """Serialise a point cloud to a compact base64 binary dict for Three.js."""
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        cols = np.full((len(pts), 3), 120, dtype=np.uint8)
    return {
        "n":       len(pts),
        "v":       version,
        "pts_b64": base64.b64encode(pts.tobytes()).decode(),
        "col_b64": base64.b64encode(cols.tobytes()).decode(),
    }


def _frames_to_rgbd(depth_frame, color_bgr: np.ndarray, depth_trunc: float):
    depth = np.frombuffer(
        depth_frame.get_buffer_as_uint16(), dtype=np.uint16, count=WIDTH * HEIGHT
    ).reshape((HEIGHT, WIDTH)).copy()
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(depth),
        depth_scale=1000.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
    )
    return rgbd, color_bgr


# ── Main SLAM loop ────────────────────────────────────────────────────────────

def run_openni_open3d_slam(
    redist_path: str,
    depth_trunc: float = 4.0,
    voxel_length: float = 0.01,
    update_every: int = 30,
    color_index: int = -1,
    save_ply: str = "slam_output.ply",
    port: int = 5000,
) -> None:
    openni2.initialize(redist_path)
    device = depth_stream = cap = None
    try:
        device = openni2.Device.open_any()
        try:
            device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        except Exception:
            pass

        depth_stream = _open_depth_stream(device)
        print("[SLAM] depth stream ok", flush=True)
        cap = _open_color_camera(color_index)
        print("[SLAM] colour camera ok", flush=True)

        intr = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, 570.0, 570.0, 320.0, 240.0)
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length, sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        odom_jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        odom_opt = o3d.pipelines.odometry.OdometryOption()

        prev_rgbd = None
        t_world_cam = np.eye(4, dtype=np.float64)
        live_pcd = o3d.geometry.PointCloud()
        frame_idx = 0

        print(f"[SLAM] Web viewer → http://127.0.0.1:{port}/", flush=True)
        print(f"[SLAM] Map refreshes every {update_every} frames. Click 'Stop & Save' in browser to finish.", flush=True)
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()

        while not _stop_evt.is_set():
            depth_frame = depth_stream.read_frame()
            ret, color_bgr = cap.read()
            if not ret or color_bgr is None:
                continue
            color_bgr = cv2.resize(color_bgr, (WIDTH, HEIGHT))

            # push depth colourmap immediately (every frame = smooth)
            _push(_depth_q, _depth_colormap(depth_frame), quality=75)

            rgbd, color_bgr = _frames_to_rgbd(depth_frame, color_bgr, depth_trunc)

            if prev_rgbd is not None:
                ok, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    prev_rgbd, rgbd, intr,
                    np.eye(4, dtype=np.float64), odom_jac, odom_opt,
                )
                if ok:
                    t_world_cam = t_world_cam @ np.linalg.inv(trans)

            volume.integrate(rgbd, intr, np.linalg.inv(t_world_cam))
            prev_rgbd = rgbd
            frame_idx += 1

            pos = t_world_cam[:3, 3]
            _stats.update(frame=frame_idx, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))

            # push RGB every frame
            _push(_rgb_q, color_bgr)

            # rebuild point cloud + update Three.js payload every N frames
            # runs in a background thread to avoid blocking the SLAM loop
            if frame_idx % update_every == 0 and _pcd_build_lock.acquire(blocking=False):
                def _rebuild(pcd_vol=volume, vl=voxel_length, fi=frame_idx):
                    global _pcd_payload
                    try:
                        pcd = pcd_vol.extract_point_cloud()
                        pcd = pcd.voxel_down_sample(voxel_size=vl * 3)   # 3× down for browser perf
                        payload = _pcd_to_payload(pcd, fi)
                        with _pcd_lock:
                            _pcd_payload = payload
                        _stats["pts"] = payload["n"]
                    finally:
                        _pcd_build_lock.release()
                threading.Thread(target=_rebuild, daemon=True).start()
                print(
                    f"[SLAM] frame={frame_idx}  cam=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})",
                    flush=True,
                )

        # ── save final PLY ────────────────────────────────────────────────
        print("[SLAM] Extracting final point cloud…", flush=True)
        final_pcd = volume.extract_point_cloud()
        final_pcd = final_pcd.voxel_down_sample(voxel_size=voxel_length)
        o3d.io.write_point_cloud(save_ply, final_pcd)
        print(f"[SLAM] Saved {len(final_pcd.points):,} pts → {save_ply}", flush=True)
        print(
            f'[SLAM] 3-D view: python -c "import open3d as o3d; '
            f"o3d.visualization.draw_geometries([o3d.io.read_point_cloud(r'{save_ply}')])\""
        )
    finally:
        if cap is not None:
            try: cap.release()
            except Exception: pass
        if depth_stream is not None:
            try: depth_stream.stop()
            except Exception: pass
        if device is not None:
            try: device.close()
            except Exception: pass
        openni2.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenNI + Open3D RGB-D SLAM with HTML web viewer")
    parser.add_argument(
        "--redist-path", type=str,
        default=(
            r"C:\Users\Krishna.Munta\Downloads\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release"
            r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
            r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
            r"\Win64-Release\sdk\libs"
        ),
    )
    parser.add_argument("--depth-trunc",   type=float, default=4.0)
    parser.add_argument("--voxel-length",  type=float, default=0.01)
    parser.add_argument("--update-every",  type=int,   default=20)
    parser.add_argument("--color-index",   type=int,   default=-1)
    parser.add_argument("--save-ply",      type=str,   default="slam_output.ply")
    parser.add_argument("--port",          type=int,   default=5000)
    args = parser.parse_args()

    redist = Path(args.redist_path)
    if not redist.exists():
        raise FileNotFoundError(f"OpenNI redist path not found: {redist}")

    # start Flask server in a daemon thread
    threading.Thread(
        target=lambda: _app.run(
            host="127.0.0.1", port=args.port,
            threaded=True, use_reloader=False,
        ),
        daemon=True,
    ).start()

    run_openni_open3d_slam(
        redist_path=str(redist),
        depth_trunc=args.depth_trunc,
        voxel_length=args.voxel_length,
        update_every=args.update_every,
        color_index=args.color_index,
        save_ply=args.save_ply,
        port=args.port,
    )


if __name__ == "__main__":
    main()
