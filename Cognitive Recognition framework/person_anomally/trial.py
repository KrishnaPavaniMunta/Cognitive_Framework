import cv2
import numpy as np
import torch
import torch.nn as nn
import json
from ultralytics import YOLO

# --- CONFIGURATION ---
CAMERA_ID = 4          # Must match what you used in Step 1
MODEL_NAME = "yolov8n-pose.pt"
CONFIG_PATH = 'threshold_config.json'
WEIGHTS_PATH = 'yolo_autoencoder.pth'
MIN_CONFIDENCE = 0.5

# --- 1D-CNN AUTOENCODER ARCHITECTURE (Must match Step 2 exactly) ---
class YOLO1DAutoencoder(nn.Module):
    def __init__(self, num_features=51):
        super(YOLO1DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3)   
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=3), 
            nn.ReLU(),
            nn.ConvTranspose1d(32, num_features, kernel_size=2, stride=2) 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def main():
    print("=== STEP 3: LIVE ANOMALY MONITORING ===")
    
    # 1. Load Configuration
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        threshold = config['anomaly_threshold']
        num_features = config['features']
        window_size = config['window_size']
        print(f"Loaded Config -> Threshold: {threshold:.2f}, Window: {window_size} frames")
    except Exception as e:
        print(f"Failed to load {CONFIG_PATH}. Did you run Step 2? Error: {e}")
        return

    # 2. Load PyTorch Model
    model = YOLO1DAutoencoder(num_features=num_features)
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval() # Set to evaluation mode (no learning)
    print("AI Brain loaded successfully.")

    # 3. Initialize YOLO and Camera
    yolo_model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(CAMERA_ID)
    
    raw_buffer = []
    last_good_coords = None

    print("\nSystem Active. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        current_frame_coords = []

        # Extract live coordinates
        if results[0].keypoints is not None and results[0].keypoints.conf is not None:
            kpts = results[0].keypoints.data
            if len(kpts) > 0:
                person_kpts = kpts[0]
                for kp in person_kpts:
                    x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
                    if conf >= MIN_CONFIDENCE:
                        current_frame_coords.extend([x, y, conf])
                    else:
                        current_frame_coords.extend([0.0, 0.0, 0.0])
                last_good_coords = current_frame_coords

        # Fallback
        if not current_frame_coords:
            if last_good_coords is not None:
                current_frame_coords = last_good_coords.copy()
            else:
                current_frame_coords = [0.0] * num_features

        raw_buffer.append(current_frame_coords)
        if len(raw_buffer) > window_size:
            raw_buffer.pop(0)

        # 4. Neural Network Inference
        if len(raw_buffer) == window_size:
            # Format data for PyTorch: (1, Features, Window_Size)
            live_window = np.array(raw_buffer).T
            live_tensor = torch.tensor(live_window, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                reconstruction = model(live_tensor)
                # Calculate live error
                mse_error = torch.mean((live_tensor - reconstruction) ** 2).item()
            
            # 5. Trigger Logic
            if mse_error > threshold:
                status_text = f"ANOMALY DETECTED! Err: {mse_error:.0f}"
                color = (0, 0, 255) # Red
                cv2.rectangle(annotated_frame, (5, 5), (annotated_frame.shape[1]-5, annotated_frame.shape[0]-5), color, 6)
            else:
                status_text = f"Normal. Err: {mse_error:.0f}"
                color = (0, 255, 0) # Green
                
            cv2.putText(annotated_frame, status_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(annotated_frame, f"Threshold: {threshold:.0f}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Live Monitor', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()