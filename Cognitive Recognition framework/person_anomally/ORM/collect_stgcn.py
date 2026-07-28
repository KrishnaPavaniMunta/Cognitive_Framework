"""
collect_stgcn.py — Step 1: Collect normal routine data in ST-GCN 4D format.

This script captures video frames from your camera, runs YOLOv8-pose to extract
17 COCO keypoints, normalizes them to be scale- and position-invariant (root-relative
+ shoulder-width scaling), and packs them into sliding windows.

Also collects spatial trajectory data (bbox center + area) for a separate
spatial anomaly detector.

Saved files:
  normal_routine_data_stgcn.npy  — shape (num_windows, 3, 30, 17)
  normal_spatial_data.npy        — shape (num_frames, 3)
"""

import cv2
import numpy as np
import signal
import sys
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
WINDOW_SIZE = 30        # 30 frames = ~1 second at 30 FPS
STRIDE = 15             # Collect a window every 15 frames (50% overlap)
TOTAL_WINDOWS = 2000    # Total windows to collect
MIN_CONFIDENCE = 0.5    # Keypoints below this confidence are zeroed out
CAMERA_ID = 4           # /dev/video0 — change if your camera is different
OUTPUT_FILE = 'normal_routine_data_stgcn.npy'
SPATIAL_OUTPUT_FILE = 'normal_spatial_data.npy'
FALLBACK_TIMEOUT = 60   # Frames before clearing buffer when person is lost (~2 sec at 30 FPS)

MODEL_NAME = "yolov8n-pose.pt"

# YOLO COCO keypoint order (17 joints):
#  0: nose          1: left_eye      2: right_eye
#  3: left_ear      4: right_ear     5: left_shoulder
#  6: right_shoulder 7: left_elbow   8: right_elbow
#  9: left_wrist    10: right_wrist  11: left_hip
# 12: right_hip     13: left_knee    14: right_knee
# 15: left_ankle    16: right_ankle

# =============================================================================
# HELPER: Scale-Normalize Keypoints (root-relative + shoulder-width)
# =============================================================================
def normalize_pose(kpts_xy):
    """
    Convert raw pixel coordinates to scale- and position-invariant form.

    Steps:
      1. Center on mid-hip (average of left_hip[11] and right_hip[12])
      2. Scale by shoulder width (distance between left_shoulder[5] and right_shoulder[6])

    Args:
        kpts_xy: (17, 2) array of (x, y) pixel coordinates

    Returns:
        normalized: (17, 2) array, or original if normalization fails
    """
    # Mid-hip as root
    hip_left = kpts_xy[11]
    hip_right = kpts_xy[12]
    if hip_left[0] == 0 and hip_left[1] == 0:
        return kpts_xy  # can't normalize
    if hip_right[0] == 0 and hip_right[1] == 0:
        return kpts_xy

    root = (hip_left + hip_right) / 2.0

    # Shoulder width as scale reference
    shoulder_left = kpts_xy[5]
    shoulder_right = kpts_xy[6]
    scale = np.linalg.norm(shoulder_right - shoulder_left)
    if scale < 1.0:
        return kpts_xy  # degenerate

    # Center and scale
    normalized = (kpts_xy - root) / scale
    return normalized


# =============================================================================
# SIGNAL HANDLER: Save partial data on Ctrl+C
# =============================================================================
dataset = []             # final list of (3, 30, 17) windows
spatial_dataset = []     # list of (cx, cy, area) per frame
raw_buffer = []          # list of (3, 17) arrays

def save_and_exit(signum=None, frame=None):
    """Save whatever data has been collected so far, then exit."""
    print("\n\nSaving collected data before exit...")
    if len(dataset) > 0:
        data = np.stack(dataset, axis=0)
        np.save(OUTPUT_FILE, data)
        print(f"  Pose data saved: {data.shape} → {OUTPUT_FILE}")
    if len(spatial_dataset) > 0:
        spatial = np.array(spatial_dataset, dtype=np.float32)
        np.save(SPATIAL_OUTPUT_FILE, spatial)
        print(f"  Spatial data saved: {spatial.shape} → {SPATIAL_OUTPUT_FILE}")
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, save_and_exit)

# =============================================================================
# INITIALIZATION
# =============================================================================
print("Loading YOLOv8-pose model...")
model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"ERROR: Could not open camera {CAMERA_ID}. Try changing CAMERA_ID.")
    exit(1)

last_good_frame = None   # fallback when no person detected
frames_since_last_window = 0
frames_since_detection = 0  # counter for fallback timeout

print(f"\n=== STEP 1: GATHERING NORMAL DATA (ST-GCN FORMAT) ===")
print(f"Target: {TOTAL_WINDOWS} windows | Window: {WINDOW_SIZE} frames | Stride: {STRIDE}")
print("Move naturally in front of the camera (walk, sit, stand, stretch, lie down).")
print("Press 'q' to stop early.  Ctrl+C will save partial data.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Camera feed failed.")
        break

    # Run YOLO pose estimation
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot() if results[0].keypoints is not None else frame

    # ------------------------------------------------------------------
    # Build a (3, 17) array for this frame: row 0=X, row 1=Y, row 2=Conf
    # Also extract spatial info: (cx, cy, bbox_area)
    # ------------------------------------------------------------------
    frame_array = np.zeros((3, 17), dtype=np.float32)
    person_detected = False
    spatial_vec = np.zeros(3, dtype=np.float32)  # (cx, cy, area)

    if (results[0].keypoints is not None and
            results[0].keypoints.conf is not None and
            results[0].keypoints.data is not None):

        kpts = results[0].keypoints.data  # shape: (num_persons, 17, 3)

        if len(kpts) > 0:
            person = kpts[0].cpu().numpy()  # (17, 3) -> (x, y, conf)

            # --- Scale-normalize X and Y coordinates ---
            raw_xy = person[:, :2].copy()  # (17, 2)
            normalized_xy = normalize_pose(raw_xy)

            for j in range(17):
                conf = person[j, 2]
                if conf >= MIN_CONFIDENCE:
                    frame_array[0, j] = normalized_xy[j, 0]  # normalized X
                    frame_array[1, j] = normalized_xy[j, 1]  # normalized Y
                    frame_array[2, j] = conf
                else:
                    frame_array[0, j] = 0.0
                    frame_array[1, j] = 0.0
                    frame_array[2, j] = 0.0

            # --- Extract spatial info from bounding box ---
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                bbox = results[0].boxes.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                spatial_vec = np.array([cx, cy, area], dtype=np.float32)

            last_good_frame = frame_array.copy()
            person_detected = True
            frames_since_detection = 0

    # Fallback: use last known pose, but only for a limited time
    if not person_detected:
        frames_since_detection += 1
        if last_good_frame is not None and frames_since_detection <= FALLBACK_TIMEOUT:
            frame_array = last_good_frame.copy()
        else:
            # Person lost for too long — clear buffer to avoid collecting bad data
            raw_buffer.clear()
            frames_since_detection = FALLBACK_TIMEOUT + 1  # prevent repeated clears

    # Add to rolling buffer
    raw_buffer.append(frame_array)

    # Keep buffer at exactly WINDOW_SIZE
    if len(raw_buffer) > WINDOW_SIZE:
        raw_buffer.pop(0)

    # ------------------------------------------------------------------
    # When buffer is full, collect a window every STRIDE frames
    # ------------------------------------------------------------------
    if len(raw_buffer) == WINDOW_SIZE:
        frames_since_last_window += 1
        if frames_since_last_window >= STRIDE:
            # Stack frames along axis 1: (3, 17) × 30 → (3, 30, 17)
            window = np.stack(raw_buffer, axis=1)  # (3, 30, 17)
            dataset.append(window)
            frames_since_last_window = 0

    # Always collect spatial data (every frame with a detection)
    if person_detected:
        spatial_dataset.append(spatial_vec)

    # Display progress
    progress = len(dataset)
    cv2.putText(annotated_frame, f"Windows: {progress}/{TOTAL_WINDOWS}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Buffer: {len(raw_buffer)}/{WINDOW_SIZE}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(annotated_frame, f"Spatial pts: {len(spatial_dataset)}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow('ST-GCN Data Collector', annotated_frame)

    # Quit conditions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nUser stopped early.")
        break
    if len(dataset) >= TOTAL_WINDOWS:
        print(f"\nTarget reached! Collected {TOTAL_WINDOWS} windows.")
        break

# =============================================================================
# SAVE
# =============================================================================
cap.release()
cv2.destroyAllWindows()

if len(dataset) == 0:
    print("ERROR: No data collected. Make sure a person is visible in the camera frame.")
    exit(1)

# Stack all windows: list of (3, 30, 17) → (N, 3, 30, 17)
data = np.stack(dataset, axis=0)
print(f"\nFinal pose data shape: {data.shape}  (Windows={data.shape[0]}, Channels={data.shape[1]}, Time={data.shape[2]}, Joints={data.shape[3]})")
print(f"Expected format: (Batch={data.shape[0]}, Channels=3, Time=30, Joints=17)")

np.save(OUTPUT_FILE, data)
print(f"Pose data saved to: {OUTPUT_FILE}")

# Save spatial trajectory data
if len(spatial_dataset) > 0:
    spatial_data = np.array(spatial_dataset, dtype=np.float32)
    print(f"Spatial data shape: {spatial_data.shape}  (Frames={spatial_data.shape[0]}, Features=3 [cx, cy, area])")
    np.save(SPATIAL_OUTPUT_FILE, spatial_data)
    print(f"Spatial data saved to: {SPATIAL_OUTPUT_FILE}")
else:
    print("WARNING: No spatial data collected.")
