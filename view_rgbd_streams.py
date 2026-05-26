"""
view_rgbd_streams.py  –  Live side-by-side viewer for Orbbec Astra
  Left  : RGB colour  (UVC via cv2.VideoCapture)
  Right : Depth       (OpenNI2 via openni package, colourised)

Run with:
  set OPENNI2_REDIST to the sdk/libs path, then:
  .orbbec-311v1/Scripts/python.exe view_rgbd_streams.py

Press  Q  to quit.
"""

import os, sys, traceback
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import cv2
from openni import openni2

OPENNI2_REDIST = os.environ.get(
    "OPENNI2_REDIST",
    r"C:\Users\Krishna.Munta\Downloads\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release"
    r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
    r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
    r"\Win64-Release\sdk\libs",
)

TARGET_W, TARGET_H = 640, 480   # display size per panel
DEPTH_MAX_MM      = 5000        # clip depth above this value (mm) before colourising


def open_astra_colour_camera() -> tuple[cv2.VideoCapture | None, bool]:
    """
    Locate and open the Orbbec Astra Pro HD Camera (not the laptop built-in cam).
    Returns (cap, found_ok).
    """
    # Never use index 0 (built-in laptop cam). Astra is usually index 1, but can shift.
    for idx in (1, 2, 3, 4):
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            print(f"[colour] Trying camera index {idx} backend={backend}...", flush=True)
            try:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    ret, frm = c.read()
                    if ret and frm is not None:
                        w, h = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[colour] Opened index={idx} backend={backend}  {w}x{h}", flush=True)
                        return c, True
                    c.release()
            except Exception as e:
                print(f"[colour] index {idx} backend {backend} error: {e}", flush=True)
    
    print("[colour] WARNING: Astra colour camera not found – showing grey placeholder", flush=True)
    return None, False

# ── depth mm → 8-bit colourised BGR ──────────────────────────────────────────
def colourise_depth(frame_data, width: int, height: int) -> np.ndarray:
    raw = np.frombuffer(frame_data, dtype=np.uint16).reshape(height, width)
    clipped = np.clip(raw, 0, DEPTH_MAX_MM).astype(np.float32)
    normalised = (clipped / DEPTH_MAX_MM * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)
    # make zero-depth (no return) black instead of red
    coloured[raw == 0] = (0, 0, 0)
    return coloured


def main():
    # ── OpenCV colour (UVC) ─── OPEN FIRST before OpenNI2 ──────────────────
    print("[colour] Trying camera index 1 (Astra)...", flush=True)
    cap, colour_ok = open_astra_colour_camera()
    
    # ── OpenNI2 depth ────────────────────────────────────────────────────────
    print("[depth] Initialising OpenNI2 from:", OPENNI2_REDIST, flush=True)
    openni2.initialize(OPENNI2_REDIST)
    dev = openni2.Device.open_any()
    info = dev.get_device_info()
    print(f"[depth] Device: {info.name}  uri: {info.uri}", flush=True)

    depth_stream = dev.create_depth_stream()
    print("[depth] Depth stream created, starting …", flush=True)
    depth_stream.start()
    print("[depth] Stream started  640×480 @ 30 fps", flush=True)

    cv2.namedWindow("RGBD Viewer  |  Q=quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGBD Viewer  |  Q=quit", TARGET_W * 2 + 10, TARGET_H + 40)

    frame_n = 0
    print("\nStreaming – press Q in the window to quit.\n")

    while True:
        # --- colour frame -------------------------------------------------
        if colour_ok:
            ret, colour_bgr = cap.read()
            if not ret:
                colour_bgr = np.zeros((TARGET_H, TARGET_W, 3), np.uint8)
        else:
            colour_bgr = np.zeros((TARGET_H, TARGET_W, 3), np.uint8)
            cv2.putText(colour_bgr, "No colour stream", (20, TARGET_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        # resize if necessary
        if colour_bgr.shape[:2] != (TARGET_H, TARGET_W):
            colour_bgr = cv2.resize(colour_bgr, (TARGET_W, TARGET_H))

        # --- depth frame --------------------------------------------------
        depth_frame = depth_stream.read_frame()
        depth_vis   = colourise_depth(
            depth_frame.get_buffer_as_uint16(), depth_frame.width, depth_frame.height
        )
        if depth_vis.shape[:2] != (TARGET_H, TARGET_W):
            depth_vis = cv2.resize(depth_vis, (TARGET_W, TARGET_H))
        depth_vis = cv2.flip(depth_vis, 1)  # flip horizontally

        # --- overlay labels -----------------------------------------------
        cv2.putText(colour_bgr, "RGB  (UVC)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(depth_vis,  "Depth (OpenNI2)  – JET colourmap",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(depth_vis,  f"frame {frame_n}",
                    (10, TARGET_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # --- side by side -------------------------------------------------
        divider   = np.full((TARGET_H, 10, 3), 40, np.uint8)   # dark grey bar
        composite = np.hstack([colour_bgr, divider, depth_vis])
        cv2.imshow("RGBD Viewer  |  Q=quit", composite)

        frame_n += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    # ── cleanup ──────────────────────────────────────────────────────────────
    print("Stopping …")
    depth_stream.stop()
    dev.close()
    openni2.unload()
    if colour_ok:
        cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
