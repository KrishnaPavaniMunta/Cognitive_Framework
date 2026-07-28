#!/usr/bin/env python3
"""
Visualize normal_routine_data.npy as pose skeleton animations.
Saves as an MP4 video showing the person's movement over time.
"""

import numpy as np
import cv2
import argparse

# COCO 17-keypoint skeleton connections (YOLOv8-pose format)
# Indices: 0=nose, 1=L-eye, 2=R-eye, 3=L-ear, 4=R-ear,
#          5=L-shoulder, 6=R-shoulder, 7=L-elbow, 8=R-elbow,
#          9=L-wrist, 10=R-wrist, 11=L-hip, 12=R-hip,
#          13=L-knee, 14=R-knee, 15=L-ankle, 16=R-ankle
POSE_CONNECTIONS = [
    (0, 1), (0, 2),           # nose -> eyes
    (1, 3), (2, 4),           # eyes -> ears
    (5, 6),                   # shoulders
    (5, 7), (7, 9),           # left arm: shoulder -> elbow -> wrist
    (6, 8), (8, 10),          # right arm: shoulder -> elbow -> wrist
    (5, 11), (6, 12),         # shoulders -> hips
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg: hip -> knee -> ankle
    (12, 14), (14, 16),       # right leg: hip -> knee -> ankle
]

# COCO-style keypoint names (17 keypoints, 0-indexed)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Colors for skeleton bones
BONE_COLOR = (0, 255, 0)      # Green
JOINT_COLOR = (0, 0, 255)     # Red
CONFIDENT_COLOR = (0, 255, 255)  # Yellow for high-confidence points
BG_COLOR = (30, 30, 30)       # Dark background


def draw_skeleton(frame, keypoints, confidences, img_w, img_h,
                  center=True, global_x_range=None, global_y_range=None):
    """Draw pose skeleton on a frame.
    
    If center=True: auto-centers and scales the person per-frame
    If center=False: uses a fixed global coordinate mapping so the person
                     stays at their original position in the camera frame
    """
    valid = confidences > 0.1
    if not np.any(valid):
        return frame

    if center:
        # Per-frame bounding box centering
        xs = keypoints[valid, 0]
        ys = keypoints[valid, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        pad = 30
        x_min = max(0, x_min - pad)
        x_max = x_max + pad
        y_min = max(0, y_min - pad)
        y_max = y_max + pad

        scale_x = (img_w - 40) / max(x_max - x_min, 1)
        scale_y = (img_h - 40) / max(y_max - y_min, 1)
        scale = min(scale_x, scale_y)

        offset_x = (img_w - (x_max - x_min) * scale) / 2 - x_min * scale
        offset_y = (img_h - (y_max - y_min) * scale) / 2 - y_min * scale

        def to_screen(kp):
            return (int(kp[0] * scale + offset_x), int(kp[1] * scale + offset_y))

    else:
        # Fixed global mapping: data coords -> screen coords
        # Preserves the person's position in the camera frame
        data_w = global_x_range[1] - global_x_range[0]
        data_h = global_y_range[1] - global_y_range[0]
        margin = 20
        scale = min((img_w - 2 * margin) / max(data_w, 1),
                     (img_h - 2 * margin) / max(data_h, 1))
        offset_x = margin - global_x_range[0] * scale
        offset_y = margin - global_y_range[0] * scale

        def to_screen(kp):
            return (int(kp[0] * scale + offset_x), int(kp[1] * scale + offset_y))

    # Draw connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue
        if confidences[start_idx] > 0.3 and confidences[end_idx] > 0.3:
            pt1 = to_screen(keypoints[start_idx])
            pt2 = to_screen(keypoints[end_idx])
            cv2.line(frame, pt1, pt2, BONE_COLOR, 2, cv2.LINE_AA)

    # Draw keypoints
    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
        if conf > 0.3:
            pt = to_screen(kp)
            color = JOINT_COLOR if conf > 0.7 else CONFIDENT_COLOR
            cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
            # Label
            cv2.putText(frame, str(i), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return frame


def create_video(data, sample_idx, output_path, fps=10, center=True,
                 global_x_range=None, global_y_range=None):
    """
    Create an MP4 video from a single sample's pose sequence.
    """
    sample = data[sample_idx]  # shape: (51, 30)
    n_timesteps = sample.shape[1]
    n_keypoints = sample.shape[0] // 3  # 17

    img_w, img_h = 800, 600

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))

    for t in range(n_timesteps):
        frame = np.full((img_h, img_w, 3), BG_COLOR, dtype=np.uint8)

        kp = np.zeros((n_keypoints, 2))
        conf = np.zeros(n_keypoints)
        for k in range(n_keypoints):
            kp[k, 0] = sample[k * 3, t]
            kp[k, 1] = sample[k * 3 + 1, t]
            conf[k] = sample[k * 3 + 2, t]

        frame = draw_skeleton(frame, kp, conf, img_w, img_h,
                              center=center,
                              global_x_range=global_x_range,
                              global_y_range=global_y_range)

        cv2.putText(frame, f"Sample {sample_idx}  |  Frame {t + 1}/{n_timesteps}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        writer.write(frame)

    writer.release()
    print(f"Video saved: {output_path}")


def create_multi_sample_video(data, start_idx, count, output_path, fps=6,
                              center=True, global_x_range=None, global_y_range=None):
    """
    Create an MP4 video showing multiple samples sequentially.
    """
    img_w, img_h = 800, 600
    n_timesteps = data.shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))

    for s in range(start_idx, start_idx + count):
        sample = data[s]
        n_keypoints = sample.shape[0] // 3

        for t in range(n_timesteps):
            frame = np.full((img_h, img_w, 3), BG_COLOR, dtype=np.uint8)

            kp = np.zeros((n_keypoints, 2))
            conf = np.zeros(n_keypoints)
            for k in range(n_keypoints):
                kp[k, 0] = sample[k * 3, t]
                kp[k, 1] = sample[k * 3 + 1, t]
                conf[k] = sample[k * 3 + 2, t]

            frame = draw_skeleton(frame, kp, conf, img_w, img_h,
                                  center=center,
                                  global_x_range=global_x_range,
                                  global_y_range=global_y_range)
            cv2.putText(frame, f"Sample {s}  |  Frame {t + 1}/{n_timesteps}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            writer.write(frame)

        # Add a spacer between samples
        spacer = np.full((img_h, img_w, 3), BG_COLOR, dtype=np.uint8)
        cv2.putText(spacer, f"--- Next: Sample {s + 1} ---",
                    (img_w // 2 - 120, img_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
        for _ in range(5):
            writer.write(spacer)

    writer.release()
    print(f"Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize pose routine data")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample index to visualize (default: 0)")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of samples to show in multi mode (default: 5)")
    parser.add_argument("--multi", action="store_true",
                        help="Show multiple samples sequentially")
    parser.add_argument("--output", type=str, default="pose_animation.mp4",
                        help="Output video path (default: pose_animation.mp4)")
    parser.add_argument("--fps", type=int, default=8,
                        help="Frames per second (default: 8)")
    parser.add_argument("--input", type=str, default="normal_routine_data.npy",
                        help="Input .npy file")
    parser.add_argument("--no-center", action="store_true",
                        help="Do NOT center the person; preserve raw camera-frame position")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    print(f"Loaded data shape: {data.shape}")
    print(f"Total samples available: {data.shape[0]}")

    center = not args.no_center

    # Compute global coordinate ranges for no-center mode
    if not center:
        # Extract all valid X and Y values across the entire dataset
        all_x, all_y = [], []
        for s in range(data.shape[0]):
            for k in range(17):
                x_vals = data[s, k * 3, :]
                y_vals = data[s, k * 3 + 1, :]
                conf_vals = data[s, k * 3 + 2, :]
                valid = conf_vals > 0.1
                all_x.extend(x_vals[valid].tolist())
                all_y.extend(y_vals[valid].tolist())
        global_x_range = (0.0, max(all_x) + 20) if all_x else (0, 640)
        global_y_range = (0.0, max(all_y) + 20) if all_y else (0, 480)
        print(f"Global X range: {global_x_range}")
        print(f"Global Y range: {global_y_range}")
        print(f"Mode: RAW positions (no centering)")
    else:
        global_x_range = None
        global_y_range = None
        print(f"Mode: Per-frame centering")

    if args.multi:
        create_multi_sample_video(data, args.sample, args.count, args.output,
                                  fps=args.fps, center=center,
                                  global_x_range=global_x_range,
                                  global_y_range=global_y_range)
    else:
        create_video(data, args.sample, args.output,
                     fps=args.fps, center=center,
                     global_x_range=global_x_range,
                     global_y_range=global_y_range)


if __name__ == "__main__":
    main()
