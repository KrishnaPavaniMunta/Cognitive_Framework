"""
trial_stgcn.py — Step 3: Live Anomaly Detection with ST-GCN Autoencoder.

Captures live camera feed, extracts YOLO pose keypoints (scale-normalized),
builds 30-frame windows in (Batch=1, Channels=3, Time=30, Joints=17) format,
and runs the trained ST-GCN autoencoder + spatial GMM to detect anomalies.

Three anomaly signals:
  1. Reconstruction error (ST-GCN) — unusual body movement
  2. Latent GMM — unusual movement pattern in latent space
  3. Spatial GMM — person in unusual location or at unusual distance

Green = Normal  |  Red border = Anomaly Detected
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import json
import os
import time
import joblib
from ultralytics import YOLO

# Import our ST-GCN autoencoder
from st_gcn import STGCNAutoencoder

# =============================================================================
# CONFIGURATION
# =============================================================================
CAMERA_ID = 0             # Match your camera device
MODEL_NAME = "yolov8n-pose.pt"
CONFIG_PATH = 'threshold_config_stgcn.json'
WEIGHTS_PATH = 'stgcn_autoencoder.pth'
MIN_CONFIDENCE = 0.5       # Keypoint confidence threshold
SMOOTHING_ALPHA = 0.3      # Exponential moving average for error display (0=no smoothing)
FALLBACK_TIMEOUT = 60      # Frames before clearing buffer when person is lost (~2 sec at 30 FPS)
ANOMALY_PERSISTENCE_SEC = 5.0  # Seconds of sustained anomaly before triggering alert

# COCO skeleton edges for privacy-mode drawing
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
    (0, 5), (0, 6),                           # neck to shoulders
]

# =============================================================================
# HELPER: Draw skeleton on a black canvas (privacy-preserving)
# =============================================================================
def draw_skeleton_on_black(kpts_xy_conf, frame_width, frame_height):
    """
    Draw only the skeleton on a pure black background — no room visible.

    Args:
        kpts_xy_conf: (17, 3) array of (x, y, confidence) in pixel coords
        frame_width, frame_height: dimensions of the output canvas

    Returns:
        black_canvas: (H, W, 3) uint8 BGR image with skeleton overlay
    """
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Draw bones
    for src, dst in SKELETON_EDGES:
        if kpts_xy_conf[src, 2] >= MIN_CONFIDENCE and kpts_xy_conf[dst, 2] >= MIN_CONFIDENCE:
            pt1 = (int(kpts_xy_conf[src, 0]), int(kpts_xy_conf[src, 1]))
            pt2 = (int(kpts_xy_conf[dst, 0]), int(kpts_xy_conf[dst, 1]))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

    # Draw joints
    for j in range(17):
        if kpts_xy_conf[j, 2] >= MIN_CONFIDENCE:
            pt = (int(kpts_xy_conf[j, 0]), int(kpts_xy_conf[j, 1]))
            cv2.circle(canvas, pt, 4, (0, 200, 255), -1)

    return canvas

# =============================================================================
# HELPER: Scale-Normalize Keypoints (root-relative + shoulder-width)
# =============================================================================
def normalize_pose(kpts_xy):
    """
    Convert raw pixel coordinates to scale- and position-invariant form.
    Centers on mid-hip, scales by shoulder width.
    """
    hip_left = kpts_xy[11]
    hip_right = kpts_xy[12]
    if (hip_left[0] == 0 and hip_left[1] == 0) or (hip_right[0] == 0 and hip_right[1] == 0):
        return kpts_xy

    root = (hip_left + hip_right) / 2.0

    shoulder_left = kpts_xy[5]
    shoulder_right = kpts_xy[6]
    scale = np.linalg.norm(shoulder_right - shoulder_left)
    if scale < 1.0:
        return kpts_xy

    return (kpts_xy - root) / scale

# =============================================================================
# LOAD CONFIG & MODEL
# =============================================================================
def load_model_and_config():
    """Load the trained ST-GCN autoencoder and threshold config."""
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: '{CONFIG_PATH}' not found. Run train_stgcn.py first.")
        return None, None, None

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: '{WEIGHTS_PATH}' not found. Run train_stgcn.py first.")
        return None, None, None

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    threshold = config['anomaly_threshold']
    num_joints = config.get('num_joints', 17)
    time_frames = config.get('time_frames', 30)
    in_features = config.get('in_features', 3)
    latent_dim = config.get('latent_dim', 64)

    print(f"Loaded config:")
    print(f"  Threshold (recon): {threshold:.6f}")
    print(f"  Joints: {num_joints} | Time: {time_frames} | Features: {in_features}")

    # Load GMM scorer
    gmm = None
    gmm_threshold = None
    if os.path.exists('gmm_scorer.pkl'):
        gmm = joblib.load('gmm_scorer.pkl')
        gmm_threshold = config.get('gmm_anomaly_threshold', None)
        if gmm_threshold is not None:
            print(f"  GMM threshold (log-prob): {gmm_threshold:.2f}")
        print(f"  GMM loaded: {gmm.n_components} components")
    else:
        print("  WARNING: gmm_scorer.pkl not found. GMM scoring disabled.")

    # Load Spatial GMM scorer
    spatial_gmm = None
    spatial_threshold = None
    if os.path.exists('gmm_spatial.pkl'):
        spatial_gmm = joblib.load('gmm_spatial.pkl')
        spatial_threshold = config.get('spatial_gmm_threshold', None)
        if spatial_threshold is not None:
            print(f"  Spatial GMM threshold (log-prob): {spatial_threshold:.2f}")
        print(f"  Spatial GMM loaded: {spatial_gmm.n_components} components")
    else:
        print("  WARNING: gmm_spatial.pkl not found. Spatial scoring disabled.")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCNAutoencoder(
        in_features=in_features,
        latent_dim=latent_dim,
        num_joints=num_joints,
        time_frames=time_frames
    ).to(device)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load normalization stats
    norm_stats = checkpoint.get('norm_stats', {
        'mean_x': 0.0, 'mean_y': 0.0,
        'std_x': 1.0, 'std_y': 1.0,
    })

    print(f"Model loaded. Device: {device}")
    return model, threshold, config, norm_stats, device, gmm, gmm_threshold, spatial_gmm, spatial_threshold


def main():
    print("=== STEP 3: LIVE ANOMALY DETECTION (ST-GCN) ===\n")

    result = load_model_and_config()
    if result[0] is None:
        return
    model, threshold, config, norm_stats, device, gmm, gmm_threshold, spatial_gmm, spatial_threshold = result

    use_gmm = (gmm is not None and gmm_threshold is not None)
    use_spatial = (spatial_gmm is not None and spatial_threshold is not None)

    # ------------------------------------------------------------------
    # Initialize YOLO + Camera
    # ------------------------------------------------------------------
    yolo_model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_ID}")
        return

    raw_buffer = []           # list of (3, 17) numpy arrays
    last_good_frame = None    # fallback
    smoothed_error = 0.0      # for display smoothing
    frames_since_detection = 0  # counter for fallback timeout
    anomaly_start_time = None   # timestamp when anomaly first detected (time-based persistence)

    print("\nSystem Active. Press 'q' to quit.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # ------------------------------------------------------------------
        # 1. YOLO Pose Extraction → (3, 17) array (scale-normalized)
        #    Also extract spatial info: (cx, cy, bbox_area)
        # ------------------------------------------------------------------
        results = yolo_model(frame, verbose=False)
        frame_height, frame_width = frame.shape[:2]

        # Privacy mode: draw skeleton on black canvas instead of showing room
        display_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        frame_array = np.zeros((3, 17), dtype=np.float32)
        person_detected = False
        spatial_vec = np.zeros(3, dtype=np.float32)  # (cx, cy, area)
        raw_kpts = None  # raw pixel coords for drawing

        if (results[0].keypoints is not None and
                results[0].keypoints.conf is not None and
                results[0].keypoints.data is not None):

            kpts = results[0].keypoints.data
            if len(kpts) > 0:
                person = kpts[0].cpu().numpy()  # (17, 3)
                raw_kpts = person.copy()  # keep raw pixels for drawing

                # --- Scale-normalize X and Y coordinates ---
                raw_xy = person[:, :2].copy()
                normalized_xy = normalize_pose(raw_xy)

                for j in range(17):
                    x, y, conf = person[j]
                    if conf >= MIN_CONFIDENCE:
                        frame_array[0, j] = normalized_xy[j, 0]
                        frame_array[1, j] = normalized_xy[j, 1]
                        frame_array[2, j] = conf
                    # else: stays 0.0

                # --- Extract spatial info from bounding box ---
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    bbox = results[0].boxes.xyxy[0].cpu().numpy()
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    spatial_vec = np.array([cx, cy, area], dtype=np.float32)

                last_good_frame = frame_array.copy()
                person_detected = True
                frames_since_detection = 0

        # Draw skeleton on black canvas (privacy-preserving)
        if raw_kpts is not None:
            display_frame = draw_skeleton_on_black(raw_kpts, frame_width, frame_height)

        # Fallback: use last known pose, but only for a limited time
        if not person_detected:
            frames_since_detection += 1
            if last_good_frame is not None and frames_since_detection <= FALLBACK_TIMEOUT:
                frame_array = last_good_frame.copy()
            else:
                # Person lost for too long — clear buffer to avoid false negatives
                raw_buffer.clear()
                smoothed_error = 0.0
                frames_since_detection = FALLBACK_TIMEOUT + 1  # prevent repeated clears

        # ------------------------------------------------------------------
        # 2. Rolling Window Management
        # ------------------------------------------------------------------
        raw_buffer.append(frame_array)
        if len(raw_buffer) > config.get('time_frames', 30):
            raw_buffer.pop(0)

        # ------------------------------------------------------------------
        # 3. ST-GCN Inference (only when buffer is full)
        # ------------------------------------------------------------------
        if len(raw_buffer) == config.get('time_frames', 30):
            # Build 4D tensor: (1, 3, 30, 17)
            window = np.stack(raw_buffer, axis=1)          # (3, 30, 17)
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 3, 30, 17)

            # Scale-normalized coords need minimal global normalization
            # (mean/std from training are near 0/1 for normalized coords)
            mean_x = norm_stats.get('mean_x', 0.0)
            mean_y = norm_stats.get('mean_y', 0.0)
            std_x = norm_stats.get('std_x', 1.0)
            std_y = norm_stats.get('std_y', 1.0)

            window_tensor[:, 0, :, :] = (window_tensor[:, 0, :, :] - mean_x) / std_x
            window_tensor[:, 1, :, :] = (window_tensor[:, 1, :, :] - mean_y) / std_y

            window_tensor = window_tensor.to(device)

            with torch.no_grad():
                reconstruction, latent = model(window_tensor)
                mse_error = torch.mean((window_tensor - reconstruction) ** 2).item()

            # Exponential moving average for smoother display
            smoothed_error = (SMOOTHING_ALPHA * mse_error +
                              (1 - SMOOTHING_ALPHA) * smoothed_error)

            # --- GMM scoring on latent space ---
            gmm_log_prob = None
            gmm_anomaly = False
            if use_gmm:
                latent_np = latent.cpu().numpy()
                gmm_log_prob = float(gmm.score_samples(latent_np)[0])
                gmm_anomaly = (gmm_log_prob < gmm_threshold)

            # --- Spatial GMM scoring ---
            spatial_log_prob = None
            spatial_anomaly = False
            if use_spatial and person_detected:
                spatial_log_prob = float(spatial_gmm.score_samples([spatial_vec])[0])
                spatial_anomaly = (spatial_log_prob < spatial_threshold)

            # ------------------------------------------------------------------
            # 4. Anomaly Decision (triple-path: recon + latent GMM + spatial)
            #    Time-based persistence: must stay anomalous for ANOMALY_PERSISTENCE_SEC
            # ------------------------------------------------------------------
            recon_anomaly = (smoothed_error > threshold)
            any_anomaly = recon_anomaly or gmm_anomaly or spatial_anomaly

            now = time.time()
            if any_anomaly:
                if anomaly_start_time is None:
                    anomaly_start_time = now
                elapsed = now - anomaly_start_time
            else:
                anomaly_start_time = None
                elapsed = 0.0

            # Only trigger alert if anomaly persists for the required duration
            alert_active = (anomaly_start_time is not None and elapsed >= ANOMALY_PERSISTENCE_SEC)

            if alert_active:
                reasons = []
                if recon_anomaly:
                    reasons.append(f"Recon: {smoothed_error:.4f}")
                if gmm_anomaly:
                    reasons.append(f"GMM: {gmm_log_prob:.2f}")
                if spatial_anomaly:
                    reasons.append(f"Spatial: {spatial_log_prob:.2f}")
                status_text = f"ANOMALY! {' | '.join(reasons)}"
                color = (0, 0, 255)  # Red
                # Draw red border
                cv2.rectangle(display_frame, (5, 5),
                              (display_frame.shape[1] - 5, display_frame.shape[0] - 5),
                              color, 6)
            elif anomaly_start_time is not None:
                # Building up — show warning with elapsed time
                status_text = f"Warning... {elapsed:.1f}s/{ANOMALY_PERSISTENCE_SEC:.0f}s  Recon: {smoothed_error:.4f}"
                color = (0, 165, 255)  # Orange
            else:
                status_text = f"Normal  Recon: {smoothed_error:.4f}"
                color = (0, 255, 0)  # Green

            # ------------------------------------------------------------------
            # 5. Display
            # ------------------------------------------------------------------
            cv2.putText(display_frame, status_text,
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(display_frame, f"Recon threshold: {threshold:.4f}",
                        (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show GMM log-prob if available
            if gmm_log_prob is not None:
                gmm_color = (0, 0, 255) if gmm_anomaly else (0, 255, 0)
                cv2.putText(display_frame, f"GMM logP: {gmm_log_prob:.2f} (thresh: {gmm_threshold:.2f})",
                            (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gmm_color, 1)

            # Show Spatial log-prob if available
            if spatial_log_prob is not None:
                s_color = (0, 0, 255) if spatial_anomaly else (0, 255, 0)
                cv2.putText(display_frame, f"Spatial logP: {spatial_log_prob:.2f} (thresh: {spatial_threshold:.2f})",
                            (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 1)

            # Latent space magnitude (optional diagnostic)
            latent_norm = torch.norm(latent).item()
            cv2.putText(display_frame, f"Latent norm: {latent_norm:.2f}",
                        (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        else:
            # ------------------------------------------------------------------
            # Warmup: buffer not yet full — show progress
            # ------------------------------------------------------------------
            warmup_text = f"Warming up... {len(raw_buffer)}/{config.get('time_frames', 30)}"
            cv2.putText(display_frame, warmup_text,
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            if frames_since_detection > 0:
                cv2.putText(display_frame, f"No person: {frames_since_detection}/{FALLBACK_TIMEOUT}",
                            (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

        # Show frame
        cv2.imshow('ST-GCN Anomaly Monitor', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring stopped.")


if __name__ == '__main__':
    main()
