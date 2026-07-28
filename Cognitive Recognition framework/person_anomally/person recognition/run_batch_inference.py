#!/usr/bin/env python3
"""
ETRI Action Classifier — Batch Inference Script
Based on the trainer's original inference code.
Usage:
    python3 run_batch_inference.py <folder_with_csvs> [--output results.csv]
    python3 run_batch_inference.py <single_csv_file>
"""

import pickle
import numpy as np
import pandas as pd
import glob
import os
import sys
import argparse
from tensorflow.keras.models import model_from_json

# ---------------------------------------------------------------------------
# 1. Load model + assets
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_inference_bundle(base_path):
    """Load Keras model, label encoder, and normalization stats."""
    print(f"Loading model from {base_path}/model.pkl ...")
    with open(f"{base_path}/model.pkl", "rb") as f:
        model_data = pickle.load(f)
    clf = model_from_json(model_data["architecture_json"])
    clf.set_weights(model_data["weights"])
    print(f"  Input shape: {model_data['input_shape']}, Classes: {model_data['num_classes']}")

    print(f"Loading label encoder from {base_path}/label_encoder.pkl ...")
    with open(f"{base_path}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    print(f"  Classes: {len(le.classes_)}")

    print(f"Loading normalization stats from {base_path}/normalization_stats.pkl ...")
    with open(f"{base_path}/normalization_stats.pkl", "rb") as f:
        norm_data = pickle.load(f)
    pos_mean, pos_std, vel_mean, vel_std = norm_data["norm_stats"]

    return clf, le, (pos_mean, pos_std, vel_mean, vel_std)


# ---------------------------------------------------------------------------
# 2. Preprocessing (identical to training pipeline)
# ---------------------------------------------------------------------------
def get_joint_columns_with_tracking():
    cols = ["frameNum"]
    for j in range(1, 26):
        cols += [f"joint{j}_3dX", f"joint{j}_3dY", f"joint{j}_3dZ", f"joint{j}_trackingState"]
    return cols

JOINT_COLS_TRACK = get_joint_columns_with_tracking()


def load_skeleton_csv_clean(path):
    df = pd.read_csv(path, usecols=JOINT_COLS_TRACK)
    df = df.sort_values("frameNum").reset_index(drop=True)

    coords_list = []
    for j in range(1, 26):
        xyz = df[[f"joint{j}_3dX", f"joint{j}_3dY", f"joint{j}_3dZ"]].values.astype(np.float32)
        tracking = df[f"joint{j}_trackingState"].values

        xyz[tracking == 0] = np.nan
        xyz_df = pd.DataFrame(xyz).ffill().bfill().fillna(0)
        coords_list.append(xyz_df.values.astype(np.float32))

    coords = np.concatenate(coords_list, axis=1)
    return coords


def normalize_skeleton(coords):
    frames = coords.shape[0]
    coords_3d = coords.reshape(frames, 25, 3)

    spine_base = coords_3d[:, 0, :]
    spine_shoulder = coords_3d[:, 20, :]

    torso_length = np.linalg.norm(spine_shoulder - spine_base, axis=1, keepdims=True)
    torso_length = np.clip(torso_length, 1e-3, None)

    centered = coords_3d - spine_base[:, np.newaxis, :]
    scaled = centered / torso_length[:, np.newaxis, :]

    return scaled.reshape(frames, 75)


def skeleton_to_sequence(coords, target_frames=64):
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
    velocity = np.diff(seq, axis=0, prepend=seq[0:1])
    return np.concatenate([seq, velocity], axis=1)


def preprocess_for_inference(csv_path, norm_stats):
    """Runs one CSV through the full pipeline, using training-set normalization stats."""
    pos_mean, pos_std, vel_mean, vel_std = norm_stats

    coords = load_skeleton_csv_clean(csv_path)
    if np.isnan(coords).any() or np.isinf(coords).any():
        return None

    coords_norm = normalize_skeleton(coords)
    seq = skeleton_to_sequence(coords_norm)
    if seq is None or np.isnan(seq).any():
        return None

    seq = add_motion_features(seq)

    # Z-score normalize using training stats
    seq_pos = (seq[:, :75] - pos_mean) / (pos_std + 1e-8)
    seq_vel = (seq[:, 75:] - vel_mean) / (vel_std + 1e-8)
    seq_norm = np.concatenate([seq_pos, seq_vel], axis=1)

    return seq_norm


# ---------------------------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------------------------
def predict_single(csv_path, clf, le, norm_stats, top_k=3):
    """Predict action for a single CSV file."""
    seq = preprocess_for_inference(csv_path, norm_stats)
    if seq is None:
        return {"file": csv_path, "error": "Preprocessing failed (bad/short clip)"}

    probs = clf.predict(seq[np.newaxis, ...], verbose=0)[0]
    top_indices = probs.argsort()[::-1][:top_k]

    predictions = [
        {"action_class": f"A{le.classes_[i]:03d}", "confidence": float(probs[i])}
        for i in top_indices
    ]

    return {"file": os.path.basename(csv_path), "predictions": predictions}


def predict_batch(folder_path, clf, le, norm_stats, output_csv=None):
    """Predict actions for all CSVs in a folder."""
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"\nFound {len(files)} CSV files to process\n")

    results = []
    for i, path in enumerate(files):
        result = predict_single(path, clf, le, norm_stats, top_k=1)
        if "error" in result:
            results.append({
                "file": result["file"],
                "predicted_action": None,
                "confidence": None,
                "status": "skipped"
            })
        else:
            top = result["predictions"][0]
            results.append({
                "file": result["file"],
                "predicted_action": top["action_class"],
                "confidence": round(top["confidence"], 4),
                "status": "ok"
            })
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(files)}...")

    results_df = pd.DataFrame(results)

    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved predictions to: {output_csv}")

    return results_df


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ETRI Action Classifier — Batch Inference")
    parser.add_argument("input_path", help="Path to a CSV file or folder of CSVs")
    parser.add_argument("--output", "-o", default="inference_results.csv", help="Output CSV path (batch mode only)")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-K predictions (single file mode)")
    parser.add_argument("--model-dir", default=SCRIPT_DIR, help="Directory containing model.pkl, label_encoder.pkl, normalization_stats.pkl")
    args = parser.parse_args()

    # Load model
    clf, le, norm_stats = load_inference_bundle(args.model_dir)
    print("Model, label encoder, and normalization stats loaded successfully!\n")

    if os.path.isfile(args.input_path):
        # Single file mode
        result = predict_single(args.input_path, clf, le, norm_stats, top_k=args.top_k)
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"File: {result['file']}")
            print(f"{'='*50}")
            for rank, pred in enumerate(result["predictions"], 1):
                bar = "█" * int(pred["confidence"] * 50)
                print(f"  #{rank}: {pred['action_class']}  |  {pred['confidence']*100:5.1f}%  {bar}")

    elif os.path.isdir(args.input_path):
        # Batch mode
        results_df = predict_batch(args.input_path, clf, le, norm_stats, output_csv=args.output)
        print(f"\n{'='*50}")
        print(f"Summary: {len(results_df)} files processed")
        ok_count = (results_df["status"] == "ok").sum()
        print(f"  Successful: {ok_count}")
        print(f"  Skipped: {len(results_df) - ok_count}")
        print(f"\nTop predictions:")
        print(results_df.head(10).to_string(index=False))

    else:
        print(f"ERROR: '{args.input_path}' is not a valid file or directory")


if __name__ == "__main__":
    main()