import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
WINDOW_SIZE = 30       # 30 frames = ~1 second of movement history
STRIDE = 15            # Collect a window every 15 frames (50% overlap max for diversity)
TOTAL_WINDOWS = 2000   # Total number of sequence windows to collect
MIN_CONFIDENCE = 0.5   # Ignore keypoints below this confidence
CAMERA_ID = 4          # Change if your camera is on a different /dev/videoX

# YOLO pose model
MODEL_NAME = "yolov8n-pose.pt"

# --- INITIALIZATION ---
model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"ERROR: Could not open camera {CAMERA_ID}. Try changing CAMERA_ID.")
    exit(1)

raw_buffer = []
dataset = []
last_good_coords = None     
frames_since_last_window = 0

print("=== STEP 1: GATHERING NORMAL DATA (HYBRID ANCHOR) ===")
print("Move naturally in front of the camera (walk, sit, stand).")
print("We are recording what 'NORMAL' looks like...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Camera feed failed.")
        break

    # Get frame dimensions for the GPS Anchor mapping
    frame_height, frame_width = frame.shape[:2]

    # YOLO inference
    results = model(frame, verbose=False)
    current_frame_coords = []

    if results[0].keypoints is not None and results[0].keypoints.conf is not None:
        annotated_frame = results[0].plot()
        kpts = results[0].keypoints.data  

        if len(kpts) > 0:
            person_kpts = kpts[0]  # First person only

            # --- HYBRID ANCHOR MATH ---
            # 1. Filter valid points to find true bounding box
            valid_x = [kp[0].item() for kp in person_kpts if kp[2].item() >= MIN_CONFIDENCE]
            valid_y = [kp[1].item() for kp in person_kpts if kp[2].item() >= MIN_CONFIDENCE]

            if len(valid_x) > 0 and len(valid_y) > 0:
                min_x, max_x = min(valid_x), max(valid_x)
                min_y, max_y = min(valid_y), max(valid_y)
                
                width = max_x - min_x
                height = max_y - min_y
                scale = max(width, height)
                if scale == 0: scale = 1.0 # Prevent division by zero error
                
                # Center pixel of the person
                center_x = min_x + (width / 2)
                center_y = min_y + (height / 2)

                # 2. Extract Normalized Posture (51 features)
                for kp in person_kpts:
                    x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
                    if conf >= MIN_CONFIDENCE:
                        # Shift to center (0,0) and scale down
                        norm_x = (x - center_x) / scale
                        norm_y = (y - center_y) / scale
                        current_frame_coords.extend([norm_x, norm_y, conf])
                    else:
                        current_frame_coords.extend([0.0, 0.0, 0.0])

                # 3. Extract Global GPS Anchor (2 features)
                # We divide by frame width/height so the anchor is 0.0 to 1.0, matching the normalized scale
                global_cx = center_x / frame_width
                global_cy = center_y / frame_height
                current_frame_coords.extend([global_cx, global_cy])

                last_good_coords = current_frame_coords
    else:
        annotated_frame = frame

    # Fallback if no person detected or no valid points found
    if not current_frame_coords:
        if last_good_coords is not None:
            current_frame_coords = last_good_coords.copy()
        else:
            current_frame_coords = [0.0] * 53  # 51 Posture + 2 Anchor = 53

    # Add to rolling buffer
    raw_buffer.append(current_frame_coords)

    if len(raw_buffer) > WINDOW_SIZE:
        raw_buffer.pop(0)

    # Collect windows with stride
    if len(raw_buffer) == WINDOW_SIZE:
        frames_since_last_window += 1
        if frames_since_last_window >= STRIDE:
            dataset.append(np.array(raw_buffer).T)
            frames_since_last_window = 0

    # Screen display
    progress = len(dataset)
    cv2.putText(annotated_frame, f"Windows Collected: {progress} / {TOTAL_WINDOWS}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Data Collector', annotated_frame)

    if len(dataset) >= TOTAL_WINDOWS:
        print("\nTarget reached! Processing and saving data...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nRecording canceled by user.")
        break

cap.release()
cv2.destroyAllWindows()

# --- SAVE DATA TO DISK ---
if len(dataset) >= TOTAL_WINDOWS:
    final_array = np.array(dataset)
    np.save('normal_routine_data.npy', final_array)
    print(f"Success! Data saved to disk as 'normal_routine_data.npy'")
    print(f"Final Data Shape: {final_array.shape} -> (Total Windows, Features, Time Steps)")
    print(f"Features breakdown: 51 Posture + 2 Anchor = 53 Features")
else:
    print("Not enough data collected to save.")