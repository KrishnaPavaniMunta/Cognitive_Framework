#!/usr/bin/env python3
"""
Inference script for ETRI Skeleton Action Classifier (ONNX).
Usage:
    python run_inference.py <path_to_skeleton_csv>
    python run_inference.py <path_to_skeleton_csv> --top-k 5
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import sys
import os

# ---------------------------------------------------------------------------
# 1. Preprocessing functions (must match training pipeline exactly)
# ---------------------------------------------------------------------------

def get_joint_columns_with_tracking():
    cols = ["frameNum"]
    for j in range(1, 26):
        cols += [f"joint{j}_3dX", f"joint{j}_3dY", f"joint{j}_3dZ", f"joint{j}_trackingState"]
    return cols

JOINT_COLS_TRACK = get_joint_columns_with_tracking()


def load_skeleton_csv_clean(path):
    """Load a skeleton CSV, handle missing tracking, return (frames, 75) float32 array."""
    df = pd.read_csv(path, usecols=JOINT_COLS_TRACK)
    df = df.sort_values("frameNum").reset_index(drop=True)

    coords_list = []
    for j in range(1, 26):
        xyz = df[[f"joint{j}_3dX", f"joint{j}_3dY", f"joint{j}_3dZ"]].values.astype(np.float32)
        tracking = df[f"joint{j}_trackingState"].values

        xyz[tracking == 0] = np.nan
        xyz_df = pd.DataFrame(xyz).ffill().bfill().fillna(0)
        coords_list.append(xyz_df.values.astype(np.float32))

    coords = np.concatenate(coords_list, axis=1)  # (frames, 75)
    return coords


def normalize_skeleton(coords):
    """Center on SpineBase (joint1), scale by torso length (SpineBase→SpineShoulder)."""
    frames = coords.shape[0]
    coords_3d = coords.reshape(frames, 25, 3)

    spine_base = coords_3d[:, 0, :]        # joint1  = SpineBase
    spine_shoulder = coords_3d[:, 20, :]   # joint21 = SpineShoulder

    torso_length = np.linalg.norm(spine_shoulder - spine_base, axis=1, keepdims=True)
    torso_length = np.clip(torso_length, 1e-3, None)

    centered = coords_3d - spine_base[:, np.newaxis, :]
    scaled = centered / torso_length[:, np.newaxis, :]

    return scaled.reshape(frames, 75)


def skeleton_to_sequence(coords, target_frames=64):
    """Resample a variable-length clip to a fixed number of frames."""
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
    """Concatenate position + velocity → 150 features."""
    velocity = np.diff(seq, axis=0, prepend=seq[0:1])
    return np.concatenate([seq, velocity], axis=1)


def normalize_features_inplace(X, norm_stats):
    """Z-score normalize using stored training stats."""
    pos_mean, pos_std, vel_mean, vel_std = norm_stats
    n_position_features = 75

    pos_view = X[:, :n_position_features]
    vel_view = X[:, n_position_features:]

    pos_view -= pos_mean
    pos_view /= (pos_std + 1e-8)
    vel_view -= vel_mean
    vel_view /= (vel_std + 1e-8)

    return X


# ---------------------------------------------------------------------------
# 2. Full preprocessing pipeline for a single CSV
# ---------------------------------------------------------------------------

def preprocess_csv(csv_path, norm_stats):
    """Run the full pipeline on a single CSV and return (1, 64, 150) array."""
    coords = load_skeleton_csv_clean(csv_path)
    if np.isnan(coords).any() or np.isinf(coords).any():
        raise ValueError(f"NaN/Inf in raw coords from {csv_path}")

    coords_norm = normalize_skeleton(coords)
    seq = skeleton_to_sequence(coords_norm)
    if seq is None:
        raise ValueError(f"Sequence too short (< 2 frames) in {csv_path}")

    seq = add_motion_features(seq).astype(np.float32)
    if np.isnan(seq).any():
        raise ValueError(f"NaN after motion features in {csv_path}")

    # Add batch dim and normalize
    X = np.expand_dims(seq, axis=0)  # (1, 64, 150)
    X = normalize_features_inplace(X, norm_stats)

    return X


# ---------------------------------------------------------------------------
# 3. Main inference
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run ETRI action classifier inference on a skeleton CSV")
    parser.add_argument("csv_path", help="Path to the skeleton CSV file")
    parser.add_argument("--model", default="etri_action_classifier.onnx", help="Path to ONNX model")
    parser.add_argument("--label-encoder", default="label_encoder.pkl", help="Path to label encoder .pkl")
    parser.add_argument("--norm-stats", default="normalization_stats.pkl", help="Path to normalization stats .pkl")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-K predictions")
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)
    le_path = os.path.join(script_dir, args.label_encoder)
    norm_path = os.path.join(script_dir, args.norm_stats)

    # --- Load assets ---
    print(f"Loading ONNX model: {model_path}")
    import onnxruntime as ort
    session = ort.InferenceSession(model_path)

    print(f"Loading label encoder: {le_path}")
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    print(f"Loading normalization stats: {norm_path}")
    with open(norm_path, "rb") as f:
        norm_data = pickle.load(f)
    norm_stats = norm_data["norm_stats"]  # (pos_mean, pos_std, vel_mean, vel_std)

    # --- Preprocess ---
    print(f"Preprocessing: {args.csv_path}")
    X = preprocess_csv(args.csv_path, norm_stats)
    print(f"Input shape: {X.shape}")  # (1, 64, 150)

    # --- Run inference ---
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    probs = session.run([output_name], {input_name: X})[0][0]  # (num_classes,)

    # --- Decode predictions ---
    top_k = min(args.top_k, len(probs))
    top_indices = np.argsort(probs)[::-1][:top_k]

    print(f"\n{'='*50}")
    print(f"Top-{top_k} Predictions for: {os.path.basename(args.csv_path)}")
    print(f"{'='*50}")
    for rank, idx in enumerate(top_indices, 1):
        class_name = le.inverse_transform([idx])[0]
        confidence = probs[idx] * 100
        bar = "█" * int(confidence / 2)
        print(f"  #{rank}: A{class_name:03d}  |  {confidence:5.1f}%  {bar}")

    print(f"\nPredicted action class: A{le.inverse_transform([top_indices[0]])[0]:03d}")
    print(f"Confidence: {probs[top_indices[0]]*100:.1f}%")


if __name__ == "__main__":
    main()