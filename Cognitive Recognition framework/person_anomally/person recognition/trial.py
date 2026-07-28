import pickle
import numpy as np
import pandas as pd
import cv2

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

from tensorflow.keras.models import model_from_json



MODEL_DIR = "etri_model_results"   # the ORIGINAL Kinect-trained bundle
WINDOW_SIZE = 64                    #make it matching the training (target_frames)
N_JOINTS = 25



JOINT_ORDER = [
    PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_SpineMid,
    PyKinectV2.JointType_Neck, PyKinectV2.JointType_Head,
    PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft,
    PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft,
    PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight,
    PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight,
    PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft,
    PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft,
    PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight,
    PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight,
    PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_HandTipLeft,
    PyKinectV2.JointType_ThumbLeft, PyKinectV2.JointType_HandTipRight,
    PyKinectV2.JointType_ThumbRight,
]

SPINE_BASE_IDX = 0
SPINE_SHOULDER_IDX = 20



# Load the ORIGINAL Kinect-trained model

def load_inference_bundle(model_dir):
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model_data = pickle.load(f)
    clf = model_from_json(model_data["architecture_json"])
    clf.set_weights(model_data["weights"])

    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    with open(f"{model_dir}/normalization_stats.pkl", "rb") as f:
        norm_data = pickle.load(f)
    norm_stats = norm_data["norm_stats"]

    return clf, le, norm_stats



# Preprocessing

def normalize_skeleton(coords):
    """coords: (frames, 75) raw 3D joint positions -> torso-relative, scale-invariant."""
    frames = coords.shape[0]
    coords_3d = coords.reshape(frames, N_JOINTS, 3)

    spine_base = coords_3d[:, SPINE_BASE_IDX, :]
    spine_shoulder = coords_3d[:, SPINE_SHOULDER_IDX, :]

    torso_length = np.linalg.norm(spine_shoulder - spine_base, axis=1, keepdims=True)
    torso_length = np.clip(torso_length, 1e-3, None)

    centered = coords_3d - spine_base[:, np.newaxis, :]
    scaled = centered / torso_length[:, np.newaxis, :]

    return scaled.reshape(frames, N_JOINTS * 3)



def add_motion_features(seq):
    velocity = np.diff(seq, axis=0, prepend=seq[0:1])
    return np.concatenate([seq, velocity], axis=1)  # (frames, 150)



def clean_frame_buffer(raw_buffer):
    """
    raw_buffer: list of (25, 4) arrays -> [x, y, z, trackingState] per joint per frame.
    Masks untracked joints (trackingState == 0) then fills gaps within the buffer,
    same philosophy as the offline CSV cleaning step.
    """
    arr = np.array(raw_buffer)  # (frames, 25, 4)
    coords = arr[:, :, :3].copy()
    tracking = arr[:, :, 3]

    coords[tracking == 0] = np.nan

    frames = coords.shape[0]
    coords_flat = coords.reshape(frames, N_JOINTS * 3)
    coords_df = pd.DataFrame(coords_flat).ffill().bfill().fillna(0)
    return coords_df.values  # (frames, 75)



def preprocess_window(raw_buffer, norm_stats):
    pos_mean, pos_std, vel_mean, vel_std = norm_stats

    coords = clean_frame_buffer(raw_buffer)      # (WINDOW_SIZE, 75)
    coords_norm = normalize_skeleton(coords)       # (WINDOW_SIZE, 75)
    seq = add_motion_features(coords_norm)         # (WINDOW_SIZE, 150)

    seq_pos = (seq[:, :75] - pos_mean) / (pos_std + 1e-8)
    seq_vel = (seq[:, 75:] - vel_mean) / (vel_std + 1e-8)
    return np.concatenate([seq_pos, seq_vel], axis=1)





# Main Kinect body-tracking loop

def run_kinect_inference():
    print("Loading model bundle from:", MODEL_DIR)
    clf, le, norm_stats = load_inference_bundle(MODEL_DIR)
    print("Loaded. Number of classes:", len(le.classes_))

    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body)

    frame_buffer = []
    current_label = "collecting frames..."
    current_confidence = 0.0

    canvas_w, canvas_h = 640, 480

    print("Starting capture. Press 'q' in the window to quit.")

    while True:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        if kinect.has_new_body_frame():
            bodies = kinect.get_last_body_frame()

            if bodies is not None:
                tracked_body = None
                for i in range(kinect.max_body_count):
                    body = bodies.bodies[i]
                    if body.is_tracked:
                        tracked_body = body
                        break

                if tracked_body is not None:
                    joints = tracked_body.joints
                    frame_joints = []
                    for joint_type in JOINT_ORDER:
                        j = joints[joint_type]
                        pos = j.Position
                        state = 2 if j.TrackingState == PyKinectV2.TrackingState_Tracked else (
                            1 if j.TrackingState == PyKinectV2.TrackingState_Inferred else 0
                        )
                        frame_joints.append([pos.x, pos.y, pos.z, state])

                        px = int(canvas_w / 2 + pos.x * 200)
                        py = int(canvas_h / 2 - pos.y * 200)
                        if 0 <= px < canvas_w and 0 <= py < canvas_h:
                            cv2.circle(canvas, (px, py), 3, (0, 255, 0), -1)

                    frame_buffer.append(np.array(frame_joints, dtype=np.float32))
                    if len(frame_buffer) > WINDOW_SIZE:
                        frame_buffer.pop(0)

                    if len(frame_buffer) == WINDOW_SIZE:
                        try:
                            seq_final = preprocess_window(frame_buffer, norm_stats)
                            if not np.isnan(seq_final).any():
                                probs = clf.predict(seq_final[np.newaxis, ...], verbose=0)[0]
                                current_label = le.classes_[probs.argmax()]
                                current_confidence = float(probs.max())
                        except Exception as e:
                            print("Inference error:", e)
                else:
                    current_label = "no person tracked"
                    current_confidence = 0.0

        cv2.putText(canvas, f"{current_label} ({current_confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"buffer: {len(frame_buffer)}/{WINDOW_SIZE}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Kinect v2 Action Recognition (press 'q' to quit)", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kinect.close()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    run_kinect_inference()