"""
train_stgcn.py — Step 2: Train the ST-GCN Autoencoder on normal routine data.

This script loads the 4D data (Batch, 3, 30, 17) collected in Step 1, trains the
ST-GCN autoencoder to reconstruct normal skeleton sequences, and establishes an
anomaly threshold based on reconstruction error.

High reconstruction error → likely anomaly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import json
import os
from sklearn.mixture import GaussianMixture
import joblib

# Import our ST-GCN autoencoder
from st_gcn import STGCNAutoencoder, build_adjacency_matrix

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = 'normal_routine_data_stgcn.npy'
SPATIAL_DATA_FILE = 'normal_spatial_data.npy'
MODEL_SAVE_PATH = 'stgcn_autoencoder.pth'
CONFIG_SAVE_PATH = 'threshold_config_stgcn.json'
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001
THRESHOLD_MULTIPLIER = 3.0  # Higher = fewer false alarms
LATENT_DIM = 64
NUM_JOINTS = 17
TIME_FRAMES = 30
IN_FEATURES = 3  # X, Y, Confidence
VAL_SPLIT = 0.2            # Fraction of data held out for validation
EARLY_STOP_PATIENCE = 10   # Stop if val loss doesn't improve for N epochs
GMM_PERCENTILE = 5.0       # Use 5th percentile of normal log-probs as GMM threshold
SPATIAL_GMM_PERCENTILE = 1.0  # Use 1st percentile for spatial GMM (stricter)
RANDOM_SEED = 42


def main():
    print("=== STEP 2: TRAINING ST-GCN AUTOENCODER ===\n")

    # ------------------------------------------------------------------
    # 0. Set random seeds for reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Load the 4D data
    # ------------------------------------------------------------------
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Cannot find '{DATA_FILE}'. Run collect_stgcn.py first.")
        return

    data = np.load(DATA_FILE)
    print(f"Loaded data shape: {data.shape}")
    print(f"  → (Windows={data.shape[0]}, Channels={data.shape[1]}, "
          f"Time={data.shape[2]}, Joints={data.shape[3]})")

    # Validate shape
    expected = (data.shape[0], 3, 30, 17)
    if data.shape != expected:
        print(f"WARNING: Expected shape {expected}, got {data.shape}")
        actual_c, actual_t, actual_j = data.shape[1], data.shape[2], data.shape[3]
        print(f"  Using actual dims: C={actual_c}, T={actual_t}, J={actual_j}")
    else:
        actual_c, actual_t, actual_j = 3, 30, 17

    # ------------------------------------------------------------------
    # 2. Normalize data per-channel (X and Y) for stable training
    # ------------------------------------------------------------------
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Channel-wise normalization (only X and Y, keep confidence in [0,1])
    mean = data_tensor.mean(dim=(0, 2, 3), keepdim=True)  # (1, 3, 1, 1)
    std = data_tensor.std(dim=(0, 2, 3), keepdim=True) + 1e-8

    # Don't normalize confidence channel too aggressively
    mean[:, 2, :, :] = 0.0
    std[:, 2, :, :] = 1.0

    data_normalized = (data_tensor - mean) / std

    print(f"Normalization — X mean: {mean[0,0,0,0]:.1f}, std: {std[0,0,0,0]:.1f}")
    print(f"Normalization — Y mean: {mean[0,1,0,0]:.1f}, std: {std[0,1,0,0]:.1f}")

    # Save normalization stats for inference
    norm_stats = {
        'mean_x': mean[0, 0, 0, 0].item(),
        'mean_y': mean[0, 1, 0, 0].item(),
        'std_x': std[0, 0, 0, 0].item(),
        'std_y': std[0, 1, 0, 0].item(),
    }

    # ------------------------------------------------------------------
    # 3. Train / Validation Split
    # ------------------------------------------------------------------
    full_dataset = TensorDataset(data_normalized, data_normalized)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train windows: {train_size} | Val windows: {val_size}")
    print(f"Train batches per epoch: {len(train_loader)}")

    # ------------------------------------------------------------------
    # 4. Initialize Model
    # ------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = STGCNAutoencoder(
        in_features=actual_c,
        latent_dim=LATENT_DIM,
        num_joints=actual_j,
        time_frames=actual_t
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ------------------------------------------------------------------
    # 5. Training Loop with Early Stopping
    # ------------------------------------------------------------------
    print("\nTraining...")
    model.train()

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            reconstruction, latent = model(batch_x)
            loss = criterion(reconstruction, batch_y)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                reconstruction, _ = model(batch_x)
                loss = criterion(reconstruction, batch_y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model checkpoint
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            marker = " *" if patience_counter == 0 else ""
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}{marker}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (best val loss at epoch {best_epoch}: {best_val_loss:.6f})")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"Training complete! Best epoch: {best_epoch} | Best val loss: {best_val_loss:.6f}")

    # ------------------------------------------------------------------
    # 6. Calculate Anomaly Threshold (on validation set — unseen data)
    # ------------------------------------------------------------------
    print("\nCalculating baseline anomaly threshold on validation set...")
    model.eval()
    errors = []

    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            reconstruction, _ = model(batch_x)
            # Per-sample MSE
            for i in range(batch_x.size(0)):
                error = torch.mean((batch_x[i] - reconstruction[i]) ** 2).item()
                errors.append(error)

    errors = np.array(errors)
    max_normal_error = float(np.max(errors))
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))

    recommended_threshold = mean_error + THRESHOLD_MULTIPLIER * std_error

    print(f"Reconstruction error stats on validation data:")
    print(f"  Mean:   {mean_error:.6f}")
    print(f"  Std:    {std_error:.6f}")
    print(f"  Max:    {max_normal_error:.6f}")
    print(f"  Recommended Threshold (mean + {THRESHOLD_MULTIPLIER}σ): {recommended_threshold:.6f}")

    # ------------------------------------------------------------------
    # 6.5. Train the Gaussian Mixture Model (GMM) on Latent Space
    # ------------------------------------------------------------------
    print("\nTraining GMM on the ST-GCN Latent Space...")
    latent_vectors = []

    # Extract latent vectors from the full dataset (train + val)
    full_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in full_loader:
            batch_x = batch_x.to(device)
            _, latent = model(batch_x)
            latent_vectors.append(latent.cpu().numpy())

    # Stack into a flat array: (N_windows, latent_dim)
    X_latent = np.vstack(latent_vectors)
    print(f"  Latent matrix shape: {X_latent.shape}")

    # Fit a GMM with 3 components (e.g., walking, sitting, standing)
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=RANDOM_SEED)
    gmm.fit(X_latent)

    # Calculate normal probability thresholds
    log_probs = gmm.score_samples(X_latent)
    min_normal_log_prob = float(np.min(log_probs))
    mean_log_prob = float(np.mean(log_probs))

    # Use percentile instead of absolute minimum for headroom
    gmm_threshold = float(np.percentile(log_probs, GMM_PERCENTILE))

    print(f"  GMM Log-Likelihoods — Mean: {mean_log_prob:.2f}, "
          f"Min: {min_normal_log_prob:.2f}, "
          f"{GMM_PERCENTILE}th percentile: {gmm_threshold:.2f}")

    # Save the GMM model alongside your PyTorch weights
    joblib.dump(gmm, 'gmm_scorer.pkl')
    print(f"  GMM saved to: gmm_scorer.pkl")

    # ------------------------------------------------------------------
    # 6.6. Train Spatial GMM on Bounding Box Trajectory
    # ------------------------------------------------------------------
    print("\nTraining Spatial GMM on bounding box trajectory...")
    spatial_gmm = None
    spatial_threshold = None

    if os.path.exists(SPATIAL_DATA_FILE):
        spatial_data = np.load(SPATIAL_DATA_FILE)
        print(f"  Spatial data shape: {spatial_data.shape}  (Frames, 3 [cx, cy, area])")

        # Fit GMM on (cx, cy, area)
        spatial_gmm = GaussianMixture(n_components=5, covariance_type='full',
                                      random_state=RANDOM_SEED)
        spatial_gmm.fit(spatial_data)

        spatial_log_probs = spatial_gmm.score_samples(spatial_data)
        spatial_threshold = float(np.percentile(spatial_log_probs, SPATIAL_GMM_PERCENTILE))

        print(f"  Spatial GMM — Mean logP: {np.mean(spatial_log_probs):.2f}, "
              f"Min: {np.min(spatial_log_probs):.2f}, "
              f"{SPATIAL_GMM_PERCENTILE}th percentile: {spatial_threshold:.2f}")

        joblib.dump(spatial_gmm, 'gmm_spatial.pkl')
        print(f"  Spatial GMM saved to: gmm_spatial.pkl")
    else:
        print(f"  WARNING: '{SPATIAL_DATA_FILE}' not found. Spatial scoring disabled.")
        print(f"  Run collect_stgcn.py again to collect spatial data.")

    # ------------------------------------------------------------------
    # 7. Save Model & Config
    # ------------------------------------------------------------------
    torch.save({
        'model_state_dict': best_model_state,
        'norm_stats': norm_stats,
        'latent_dim': LATENT_DIM,
        'num_joints': actual_j,
        'time_frames': actual_t,
        'in_features': actual_c,
    }, MODEL_SAVE_PATH)

    config = {
        'mean_error': mean_error,
        'std_error': std_error,
        'max_normal_error': max_normal_error,
        'threshold_multiplier': THRESHOLD_MULTIPLIER,
        'anomaly_threshold': recommended_threshold,
        'num_joints': actual_j,
        'time_frames': actual_t,
        'in_features': actual_c,
        'latent_dim': LATENT_DIM,
        'gmm_anomaly_threshold': gmm_threshold,
        'gmm_mean_log_prob': mean_log_prob,
        'gmm_min_log_prob': min_normal_log_prob,
        'gmm_percentile': GMM_PERCENTILE,
        'spatial_gmm_threshold': spatial_threshold,
        'spatial_gmm_percentile': SPATIAL_GMM_PERCENTILE,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
    }
    config.update(norm_stats)

    with open(CONFIG_SAVE_PATH, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nModel saved to:       {MODEL_SAVE_PATH}")
    print(f"Config saved to:      {CONFIG_SAVE_PATH}")
    print(f"Anomaly threshold:    {recommended_threshold:.6f}")
    print(f"GMM threshold:        {gmm_threshold:.2f} ({GMM_PERCENTILE}th percentile)")
    if spatial_threshold is not None:
        print(f"Spatial GMM threshold: {spatial_threshold:.2f} ({SPATIAL_GMM_PERCENTILE}th percentile)")
    print("\nDone! Run trial_stgcn.py to start live anomaly monitoring.")


if __name__ == '__main__':
    main()
