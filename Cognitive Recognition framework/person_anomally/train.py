import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os

# --- CONFIGURATION ---
DATA_FILE = 'normal_routine_data.npy'
MODEL_SAVE_PATH = 'yolo_autoencoder.pth'
CONFIG_SAVE_PATH = 'threshold_config.json'
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
THRESHOLD_MULTIPLIER = 3.0  # How strict the alarm is (higher = fewer false alarms)

# --- 1D-CNN AUTOENCODER ARCHITECTURE ---
class YOLO1DAutoencoder(nn.Module):
    def __init__(self, num_features=53):
        super(YOLO1DAutoencoder, self).__init__()
        # Encoder: Compresses the 30-frame sequence
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # (30 -> 15)
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3)   # (15 -> 5) - The Bottleneck
        )
        # Decoder: Reconstructs the 30-frame sequence
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=3), # (5 -> 15)
            nn.ReLU(),
            nn.ConvTranspose1d(32, num_features, kernel_size=2, stride=2) # (15 -> 30)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    print("=== STEP 2: TRAINING THE AI MODEL ===")
    
    # 1. Load the dataset
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Cannot find {DATA_FILE}. Did you run Step 1?")
        return
        
    print(f"Loading data from {DATA_FILE}...")
    data = np.load(DATA_FILE)
    print(f"Loaded Shape: {data.shape}")
    
    # 2. Convert to PyTorch Tensors and create a DataLoader
    # Data is already in shape (Windows, Features, Time) -> (2000, 51, 30)
    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor_data, tensor_data) # Input and Target are the same
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model, Loss Function, and Optimizer
    num_features = data.shape[1] # Should be 51
    model = YOLO1DAutoencoder(num_features=num_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("\nStarting Training Phase...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            
            # Calculate how badly it reconstructed the data
            loss = criterion(outputs, batch_y)
            
            # Backpropagation (Learning)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Average MSE Loss: {avg_loss:.6f}")

    print("\nTraining Complete!")

    # 5. Evaluate the Baseline to establish the Anomaly Threshold
    print("\nCalculating Baseline Anomaly Threshold...")
    model.eval()
    max_normal_error = 0.0
    
    # We test on the full dataset without shuffling to find the highest error it naturally makes
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            reconstruction = model(batch_x)
            # Calculate MSE for this specific 1-second window
            error = torch.mean((batch_x - reconstruction) ** 2).item()
            if error > max_normal_error:
                max_normal_error = error

    recommended_threshold = max_normal_error * THRESHOLD_MULTIPLIER

    print(f"Max Normal Error Found: {max_normal_error:.6f}")
    print(f"Recommended Threshold (Max * {THRESHOLD_MULTIPLIER}): {recommended_threshold:.6f}")

    # 6. Save Model and Configuration
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    config = {
        "max_normal_error": max_normal_error,
        "threshold_multiplier": THRESHOLD_MULTIPLIER,
        "anomaly_threshold": recommended_threshold,
        "features": num_features,
        "window_size": data.shape[2]
    }
    
    with open(CONFIG_SAVE_PATH, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"\nModel weights saved to '{MODEL_SAVE_PATH}'")
    print(f"Configuration saved to '{CONFIG_SAVE_PATH}'")
    print("Ready for Step 3: Live Inference!")

if __name__ == "__main__":
    main()