import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import pickle
import os
from collections import deque


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "etri_action_classifier.onnx")
LE_PATH = os.path.join(SCRIPT_DIR, "label_encoder.pkl")
NORM_PATH = os.path.join(SCRIPT_DIR, "normalization_stats.pkl")

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
TARGET_FRAMES = 64
BUFFER_SIZE = 48
INFER_EVERY_N = 6
EMA_ALPHA = 0.3

# ---------------------------------------------------------------------------
# Load model and metadata
# ---------------------------------------------------------------------------
print("Loading ONNX model...", end=" ", flush=True)
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("OK")

print("Loading label encoder...", end=" ", flush=True)
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)
print(f"OK ({len(le.classes_)} classes)")

print("Loading normalization stats...", end=" ", flush=True)
with open(NORM_PATH, "rb") as f:
    norm_data = pickle.load(f)
norm_stats = norm_data["norm_stats"]
print("OK")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
# remapping to ETRI 25-joint format
def mediapipe_to_etri_25(landmarks_3d):
    """Convert MediaPipe world-space landmarks to ETRI Kinect 25-joint format."""

    def pt(i):
        return np.array([landmarks_3d[i].x, landmarks_3d[i].y, landmarks_3d[i].z])

    def mid(i, j):
        return (pt(i) + pt(j)) / 2.0

    spine_base = mid(23, 24)
    spine_shoulder = mid(11, 12)

    joints = np.zeros((25, 3), dtype=np.float32)

    joints[0] = spine_base
    joints[1] = (spine_base + spine_shoulder) / 2.0
    joints[2] = spine_shoulder + (pt(0) - spine_shoulder) * 0.5
    joints[3] = pt(0)

    joints[4] = pt(11)
    joints[5] = pt(13)
    joints[6] = pt(15)
    joints[7] = pt(19)

    joints[8] = pt(12)
    joints[9] = pt(14)
    joints[10] = pt(16)
    joints[11] = pt(20)

    joints[12] = pt(23)
    joints[13] = pt(25)
    joints[14] = pt(27)
    joints[15] = pt(31)

    joints[16] = pt(24)
    joints[17] = pt(26)
    joints[18] = pt(28)
    joints[19] = pt(32)

    joints[20] = spine_shoulder
    joints[21] = pt(19)
    joints[22] = pt(21)
    joints[23] = pt(20)
    joints[24] = pt(22)

    return joints



def normalize_skeleton(coords_3d):
    """Center on SpineBase, scale by torso length (SpineBase to SpineShoulder)."""
    frames = coords_3d.shape[0]
    spine_base = coords_3d[:, 0, :]
    spine_shoulder = coords_3d[:, 20, :]

    torso_length = np.linalg.norm(spine_shoulder - spine_base, axis=1, keepdims=True)
    torso_length = np.clip(torso_length, 1e-3, None)

    centered = coords_3d - spine_base[:, np.newaxis, :]
    scaled = centered / torso_length[:, np.newaxis, :]

    return scaled.reshape(frames, 75)


def skeleton_to_sequence(coords, target_frames=TARGET_FRAMES):
    """Resample variable-length clip to fixed frame count via linear interpolation."""
    frames, n_features = coords.shape
    if frames < 2:
        return None
    old_idx = np.linspace(0, frames - 1, frames)
    new_idx = np.linspace(0, frames - 1, target_frames)
    resampled = np.zeros((target_frames, n_features), dtype=np.float32)
    for f in range(n_features):
        resampled[:, f] = np.interp(new_idx, old_idx, coords[:, f])
    return resampled


def add_motion_features(seq):
    """Concatenate position and velocity to produce 150 features."""
    velocity = np.diff(seq, axis=0, prepend=seq[0:1])
    return np.concatenate([seq, velocity], axis=1)


def normalize_features_inplace(X, norm_stats):
    """Z-score normalize using pre-computed training statistics."""
    pos_mean, pos_std, vel_mean, vel_std = norm_stats
    pos_view = X[:, :75]
    vel_view = X[:, 75:]
    pos_view -= pos_mean
    pos_view /= (pos_std + 1e-8)
    vel_view -= vel_mean
    vel_view /= (vel_std + 1e-8)
    return X


def preprocess_buffer(joints_buffer):
    """Convert a deque of (25, 3) joint arrays into model-ready (1, 64, 150) tensor."""
    coords_3d = np.stack(joints_buffer, axis=0)
    coords_flat = normalize_skeleton(coords_3d)
    seq = skeleton_to_sequence(coords_flat, TARGET_FRAMES)
    seq = add_motion_features(seq).astype(np.float32)
    X = np.expand_dims(seq, axis=0)
    X = normalize_features_inplace(X, norm_stats)
    return X
    
#def main():
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    joints_buffer = deque(maxlen=BUFFER_SIZE)
    last_prediction = "Waiting..."
    last_confidence = 0.0
    top_predictions = []
    frame_count = 0
    ema_probs = None

    print("\nWebcam running - press 'q' to quit.\n")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        h, w, _ = frame.shape

        if results.pose_world_landmarks:
            etri_joints = mediapipe_to_etri_25(results.pose_world_landmarks.landmark)
            joints_buffer.append(etri_joints)

            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

        frame_count += 1

        if frame_count % INFER_EVERY_N == 0 and len(joints_buffer) >= 16:
            try:
                X = preprocess_buffer(list(joints_buffer))
                probs = session.run([output_name], {input_name: X})[0][0]

                if ema_probs is None:
                    ema_probs = probs.copy()
                else:
                    ema_probs = EMA_ALPHA * probs + (1 - EMA_ALPHA) * ema_probs

                top_k = 3
                top_indices = np.argsort(ema_probs)[::-1][:top_k]
                top_predictions = [
                    (le.inverse_transform([idx])[0], ema_probs[idx] * 100)
                    for idx in top_indices
                ]
                last_prediction = f"A{top_predictions[0][0]:03d}"
                last_confidence = top_predictions[0][1]

                if frame_count % (INFER_EVERY_N * 5) == 0:
                    preds_str = " | ".join(
                        f"A{cls:03d}: {conf:5.1f}%" for cls, conf in top_predictions
                    )
                    print(f"\r  [{last_prediction}] {preds_str}   ", end="", flush=True)

            except Exception as e:
                print(f"\n  Inference error: {e}")

        # --- HUD overlay ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        cv2.putText(frame, f"Action: {last_prediction}",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {last_confidence:.1f}%",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        y_offset = 120
        for i, (cls_name, conf) in enumerate(top_predictions):
            label = f"A{cls_name:03d}"
            bar_len = int(conf * 2)
            cv2.putText(frame, label,
                        (15, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.rectangle(frame, (100, y_offset + i * 25 - 12),
                          (100 + bar_len, y_offset + i * 25 - 2), (0, 255, 0), -1)
            cv2.putText(frame, f"{conf:.1f}%",
                        (105 + bar_len, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(frame, f"Buffer: {len(joints_buffer)}/{BUFFER_SIZE} frames",
                    (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("ETRI Action Recognition - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\nDone.")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()