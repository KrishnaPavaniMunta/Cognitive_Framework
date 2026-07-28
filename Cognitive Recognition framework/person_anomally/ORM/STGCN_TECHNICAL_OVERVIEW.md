# ST-GCN Anomaly Detection — Technical Overview

> A deep-dive into the architecture, mathematics, and design decisions behind the skeleton-based anomaly detection system.

---

## Table of Contents

1. [Data Representation](#1-data-representation)
2. [Pose Preprocessing Pipeline](#2-pose-preprocessing-pipeline)
3. [Graph Construction — The Adjacency Matrix](#3-graph-construction--the-adjacency-matrix)
4. [ST-GCN Block — The Core Building Block](#4-st-gcn-block--the-core-building-block)
5. [Encoder: STGCNEncoder](#5-encoder-stgcnencoder)
6. [Decoder: STGCNDecoder](#6-decoder-stgcndecoder)
7. [Full Autoencoder: STGCNAutoencoder](#7-full-autoencoder-stgcnautoencoder)
8. [Training Procedure](#8-training-procedure)
9. [Anomaly Detection — Three Independent Signals](#9-anomaly-detection--three-independent-signals)
10. [Inference Pipeline (Real-time)](#10-inference-pipeline-real-time)
11. [Key Architectural Decisions & Justification](#11-key-architectural-decisions--their-justification)
12. [Files & Data Flow](#12-files--data-flow)

---

## 1. Data Representation

The system operates on **4D tensors** of shape `(Batch, Channels, Time, Joints)`:

| Dimension | Size | Meaning |
|---|---|---|
| $N$ (Batch) | Variable | Number of independent 1-second windows |
| $C$ (Channels) | 3 | Normalized $X$, normalized $Y$, confidence $\in [0,1]$ |
| $T$ (Time) | 30 | Frames (~1 second at 30 FPS) |
| $V$ (Joints) | 17 | COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |

### COCO Keypoint Indices

| Index | Joint | Index | Joint |
|---|---|---|---|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

---

## 2. Pose Preprocessing Pipeline

### 2.1 Extraction

**YOLOv8-nano-pose** detects the person and outputs 17 keypoints per frame as `(x, y, confidence)` in pixel coordinates.

### 2.2 Scale-Position Normalization

The raw pixel coordinates are normalized to be invariant to camera distance and person position:

$$\text{root} = \frac{\text{hip}_\text{left} + \text{hip}_\text{right}}{2}$$

$$\text{scale} = \|\text{shoulder}_\text{right} - \text{shoulder}_\text{left}\|_2$$

$$\mathbf{kpts}_{\text{norm}} = \frac{\mathbf{kpts} - \text{root}}{\text{scale}}$$

This makes the representation **invariant** to:
- **Camera distance** — normalized by shoulder width
- **Person position in frame** — centered on mid-hip

If normalization fails (hips not detected, shoulder width near zero), the raw coordinates are used as a fallback.

### 2.3 Channel-wise Standardization (Z-score)

During training, per-channel mean $\mu$ and standard deviation $\sigma$ are computed across all windows, frames, and joints:

$$\mathbf{X}_{\text{final}}[:, 0, :, :] = \frac{\mathbf{X}_{\text{norm}}[:, 0, :, :] - \mu_x}{\sigma_x} \quad \text{(X channel)}$$

$$\mathbf{X}_{\text{final}}[:, 1, :, :] = \frac{\mathbf{X}_{\text{norm}}[:, 1, :, :] - \mu_y}{\sigma_y} \quad \text{(Y channel)}$$

The confidence channel is left untouched ($\mu=0, \sigma=1$) as it already lies in $[0, 1]$.

---

## 3. Graph Construction — The Adjacency Matrix

The human skeleton is modeled as an **undirected graph** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with $|\mathcal{V}| = 17$ nodes (joints) and $|\mathcal{E}| = 18$ edges (bones):

```
Face:     nose ↔ left_eye, nose ↔ right_eye, left_eye ↔ left_ear, right_eye ↔ right_ear
Shoulders: left_shoulder ↔ right_shoulder
Left Arm:  left_shoulder ↔ left_elbow, left_elbow ↔ left_wrist
Right Arm: right_shoulder ↔ right_elbow, right_elbow ↔ right_wrist
Torso:     left_shoulder ↔ left_hip, right_shoulder ↔ right_hip, left_hip ↔ right_hip
Left Leg:  left_hip ↔ left_knee, left_knee ↔ left_ankle
Right Leg: right_hip ↔ right_knee, right_knee ↔ right_ankle
Neck:      nose ↔ left_shoulder, nose ↔ right_shoulder
```

The raw adjacency matrix $\mathbf{A}_{\text{raw}} \in \{0,1\}^{17 \times 17}$ is built by setting $\mathbf{A}_{ij} = 1$ for each edge $(i,j) \in \mathcal{E}$ (symmetric, undirected) plus self-loops $\mathbf{A}_{ii} = 1$.

### Symmetric Normalization (Kipf & Welling, 2017)

$$\mathbf{A} = \mathbf{D}^{-\frac{1}{2}} \mathbf{A}_{\text{raw}} \mathbf{D}^{-\frac{1}{2}}, \quad \text{where } D_{ii} = \sum_j A_{\text{raw}, ij}$$

This prevents feature magnitude explosion in deep graph networks. $\mathbf{A}$ is **fixed** (not learned) and stored as a PyTorch `register_buffer` so it automatically moves to the correct device.

---

## 4. ST-GCN Block — The Core Building Block

Each `STGCNBlock` applies two operations sequentially with a residual connection:

```
        ┌──────────────────────────────────┐
        │          x (n, c_in, t, v)        │
        └──────────────┬───────────────────┘
                       │
        ┌──────────────▼───────────────────┐
        │  Spatial GCN:                    │
        │    x = einsum('nctv,vw->nctw',x,A)│  ← message passing along bones
        │    x = Conv2d(1×1)(x)             │  ← channel mixing per joint
        │    x = BatchNorm2d(x)             │
        │    x = ReLU(x)                    │
        │    x = Dropout(x)                 │
        └──────────────┬───────────────────┘
                       │
        ┌──────────────▼───────────────────┐
        │  Temporal Conv:                  │
        │    x = Conv2d(kernel=(3,1))(x)    │  ← 1D conv along time axis
        │    x = BatchNorm2d(x)             │
        └──────────────┬───────────────────┘
                       │
        ┌──────────────▼───────────────────┐
        │  Residual: x += identity          │
        │    (1×1 conv if channels differ)  │
        │  x = ReLU(x)                      │
        │  x = Dropout(x)                   │
        └──────────────┬───────────────────┘
                       │
                       ▼
                 (n, c_out, t, v)
```

### Spatial GCN: Einstein Summation

The operation `einsum('nctv,vw->nctw', x, A)` performs a batched matrix multiplication along the joint dimension while keeping batch, channel, and time dimensions independent:

$$x_{n,c,t,w}^{\text{out}} = \sum_{v=1}^{17} x_{n,c,t,v} \cdot A_{v,w}$$

This means each joint's feature vector is updated as a weighted sum of its neighbors' features, where the weights come from the normalized adjacency matrix.

### Temporal Convolution

`Conv2d(kernel_size=(3, 1), padding=(1, 0))` applies a 1D convolution along the time axis across 3 consecutive frames for each joint independently. This captures motion dynamics (velocity, acceleration) without mixing joints.

### Residual Connection

When $C_{\text{in}} \neq C_{\text{out}}$, the identity path uses a $1 \times 1$ convolution to project channels. Otherwise it's a pure identity skip. This follows the ResNet principle (He et al., 2016).

---

## 5. Encoder: STGCNEncoder

```
Input: (n, 3, 30, 17)
  │
  ├─ STGCNBlock(3 → 16)    ──► (n, 16, 30, 17)
  ├─ STGCNBlock(16 → 32)   ──► (n, 32, 30, 17)
  ├─ STGCNBlock(32 → 64)   ──► (n, 64, 30, 17)
  │
  ├─ AdaptiveAvgPool2d((1,1)) ──► (n, 64, 1, 1)   ← collapse T×V → single scalar per channel
  ├─ Flatten                  ──► (n, 64)
  └─ Linear(64 → 64)          ──► (n, 64)  latent vector z
```

**Channel progression:** $3 \rightarrow 16 \rightarrow 32 \rightarrow 64$ — progressive feature abstraction while preserving $T$ and $V$ dimensions until the final pool.

**Global Average Pooling** (`AdaptiveAvgPool2d((1,1))`) collapses all spatial-temporal information into a single 64-dimensional vector, making the latent representation independent of input sequence length. This is crucial — it means the encoder could theoretically handle variable-length sequences without architectural changes.

---

## 6. Decoder: STGCNDecoder

```
Input: z (n, 64)
  │
  ├─ Linear(64 → 64×8×5)          ──► (n, 2560)
  ├─ Reshape                      ──► (n, 64, 8, 5)
  │
  ├─ ConvTranspose2d(64 → 32, k=(4,3), s=(2,2), p=(1,1))
  │   + BN + ReLU                 ──► (n, 32, 16, 9)
  │
  ├─ ConvTranspose2d(32 → 16, k=(4,3), s=(2,2), p=(1,1))
  │   + BN + ReLU                 ──► (n, 16, 32, 17)
  │
  ├─ Conv2d(16 → 3, 1×1)          ──► (n, 3, 32, 17)   ← project to output channels
  ├─ einsum('nctv,vw→nctw', x, A)                        ← final graph refinement
  └─ AdaptiveAvgPool2d((30,17))   ──► (n, 3, 30, 17)    ← squeeze time 32 → 30
```

### Upsampling Path

The decoder uses **transposed convolutions** (fractionally-strided convolutions) to progressively expand the compressed representation:

$$(64, 8, 5) \xrightarrow{\times 2} (32, 16, 9) \xrightarrow{\times 2} (16, 32, 17)$$

### ConvTranspose2d Output Size Formula

$$H_{\text{out}} = (H_{\text{in}} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel} + \text{output\_padding}$$

For stage 1 (64 → 32):
$$H_{\text{out}} = (8 - 1) \times 2 - 2 \times 1 + 4 + 0 = 16$$
$$W_{\text{out}} = (5 - 1) \times 2 - 2 \times 1 + 3 + 0 = 9$$

### Final Graph Refinement

The `einsum('nctv,vw→nctw', x, A)` after the final projection ensures the output respects skeletal connectivity — each joint's reconstruction is influenced by its neighbors, maintaining anatomical consistency.

### Time Adjustment

The transposed convolutions produce $T=32$ time steps, but we need $T=30$. `AdaptiveAvgPool2d((30,17))` smoothly interpolates down by ~6% — negligible information loss.

---

## 7. Full Autoencoder: STGCNAutoencoder

```python
class STGCNAutoencoder(nn.Module):
    def __init__(self, in_features=3, latent_dim=64, num_joints=17, time_frames=30):
        super().__init__()
        A = build_adjacency_matrix(num_joints=num_joints)
        self.register_buffer('A', A)  # not learned, moves with .to(device)
        self.encoder = STGCNEncoder(in_features, latent_dim)
        self.decoder = STGCNDecoder(latent_dim, in_features, num_joints, time_frames)

    def forward(self, x):
        z = self.encoder(x, self.A)          # (n, 3, 30, 17) → (n, 64)
        x_hat = self.decoder(z, self.A)       # (n, 64) → (n, 3, 30, 17)
        return x_hat, z
```

### Approximate Parameter Count

| Component | Parameters |
|---|---|
| STGCNBlock(3 → 16) | ~1,500 |
| STGCNBlock(16 → 32) | ~12,000 |
| STGCNBlock(32 → 64) | ~51,000 |
| Encoder FC (64 → 64) | ~4,200 |
| Decoder FC (64 → 2,560) | ~165,000 |
| ConvTranspose2d (64 → 32) | ~105,000 |
| ConvTranspose2d (32 → 16) | ~105,000 |
| Final Conv2d (16 → 3) | ~200 |
| **Total** | **~440,000** |

Very lightweight — runs in real-time on CPU.

---

## 8. Training Procedure

### 8.1 Objective

The model is trained to minimize reconstruction error **only on normal data** — it never sees anomalies:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N \cdot C \cdot T \cdot V} \sum_{i=1}^{N} \|\mathbf{X}_i - \hat{\mathbf{X}}_i\|_2^2$$

where $N$ is batch size, and $\hat{\mathbf{X}}_i = \text{Decoder}(\text{Encoder}(\mathbf{X}_i))$.

### 8.2 Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Optimizer** | Adam ($\beta_1=0.9$, $\beta_2=0.999$) | Standard for autoencoders; adaptive learning rates |
| **Learning rate** | $10^{-3}$ | Moderate — stable convergence on normalized data |
| **LR schedule** | Cosine annealing ($T_{\max}=60$) | Smooth decay to zero without sudden drops |
| **Weight decay** | $10^{-5}$ | Light L2 regularization to prevent overfitting |
| **Batch size** | 64 | Balances GPU memory usage and gradient stability |
| **Max epochs** | 60 | With early stopping, rarely reaches this |
| **Train/Val split** | 80/20 | Threshold computed on held-out data (unseen normal) |
| **Random seed** | 42 | Reproducibility across runs |

### 8.3 Early Stopping

Training halts if validation loss doesn't improve for 10 consecutive epochs. The best model (by validation loss) is restored:

```python
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    patience_counter = 0
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
else:
    patience_counter += 1

if patience_counter >= 10:
    break  # stop training, restore best model
```

This prevents overfitting — the model stops learning once it starts memorizing training data noise rather than generalizing.

---

## 9. Anomaly Detection — Three Independent Signals

### Signal 1: Reconstruction Error (MSE)

For a test window $\mathbf{X}_{\text{test}}$:

$$\text{MSE} = \frac{1}{C \cdot T \cdot V} \|\mathbf{X}_{\text{test}} - \hat{\mathbf{X}}_{\text{test}}\|_2^2$$

$$\text{Anomaly if: } \text{MSE} > \mu_{\text{val}} + 3\sigma_{\text{val}}$$

The threshold is computed on the **validation set** (unseen normal data). The $3\sigma$ multiplier controls sensitivity:

| Multiplier | Behavior |
|---|---|
| 2.0 | More sensitive — fewer false negatives, more false positives |
| 3.0 (default) | Balanced — catches obvious anomalies, tolerates normal variation |
| 4.0 | Conservative — only flags extreme deviations |

#### Exponential Moving Average Smoothing

In live inference, raw frame-by-frame MSE is noisy. An EMA smooths the display:

$$\text{MSE}_{\text{smooth}}^{(t)} = \alpha \cdot \text{MSE}^{(t)} + (1-\alpha) \cdot \text{MSE}_{\text{smooth}}^{(t-1)}, \quad \alpha = 0.3$$

This adds negligible latency while removing jitter.

---

### Signal 2: Latent GMM (Density Estimation)

A 3-component Gaussian Mixture Model is fit on all latent vectors $\mathbf{z} \in \mathbb{R}^{64}$ from normal training data:

$$p(\mathbf{z}) = \sum_{k=1}^{3} \pi_k \cdot \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where:
- $\pi_k$ — mixing coefficient (sums to 1)
- $\boldsymbol{\mu}_k \in \mathbb{R}^{64}$ — mean of component $k$
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{64 \times 64}$ — full covariance matrix of component $k$

$$\text{Anomaly if: } \log p(\mathbf{z}_{\text{test}}) < \text{percentile}_{5\%}(\log p(\mathbf{z}_{\text{train}}))$$

**Why 3 components?** Different movement modes (walking, standing, sitting) naturally form clusters in latent space. Each Gaussian component captures one mode.

**Why 5th percentile, not min?** The absolute minimum log-probability in the training set could be an outlier itself. Using the 5th percentile provides a safety margin.

**What this catches that MSE misses:** A movement that the autoencoder reconstructs reasonably well (moderate MSE) but whose latent representation falls in a region far from any normal pattern. Example: walking normally but while holding arms in an unusual position.

---

### Signal 3: Spatial GMM (Location/Distance)

A 5-component GMM is fit on 3-dimensional vectors $[cx, cy, area]$ extracted from the person's bounding box:

| Feature | Meaning |
|---|---|
| `cx` | Horizontal center of bounding box (pixels) |
| `cy` | Vertical center of bounding box (pixels) |
| `area` | Bounding box area = width × height (pixels²) |

$$\text{Anomaly if: } \log p([cx, cy, area]_{\text{test}}) < \text{percentile}_{1\%}(\text{train})$$

**Why 5 components?** Spatial position has more natural variation than movement patterns — different corners of the room, different distances from camera.

**Why 1st percentile (stricter)?** Spatial location should be highly constrained during normal routine. A person simply shouldn't be in unexpected places. The stricter threshold reduces false negatives for zone-based anomalies.

**What this catches:** Person entering restricted areas, climbing on furniture, hiding in corners, approaching too close to the camera.

---

### Fusion Logic

$$\text{Alert} = \begin{cases} 
\text{True} & \text{if } (\text{MSE\_alert} \lor \text{GMM\_alert} \lor \text{Spatial\_alert}) \land (\Delta t \geq 5\text{s}) \\
\text{False} & \text{otherwise}
\end{cases}$$

Where $\Delta t$ is the duration of continuous anomaly since the first anomalous frame.

#### 5-Second Persistence Filter

| Duration | Display | Meaning |
|---|---|---|
| 0–5s | 🟠 Orange "Warning..." | Building up — not yet confirmed |
| >5s | 🔴 Red "ANOMALY!" + reasons | Confirmed alert |

This filters out:
- Momentary pose estimation errors (keypoint flicker)
- Brief unusual movements (scratching head, adjusting clothes)
- Single-frame detection noise

---

## 10. Inference Pipeline (Real-time)

```
Frame[t] → YOLOv8-pose → normalize_pose() → buffer.append(kpts)
                                                    │
                                          buffer.len() == 30?
                                                    │ Yes
                                                    ▼
                                    stack → (1, 3, 30, 17) tensor
                                                    │
                                          apply norm_stats
                                          (mean/std from training)
                                                    │
                                                    ▼
                                    model.forward() ──► x_hat, z
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                                  MSE           GMM(z)         GMM(bbox)
                                    │               │               │
                                    ▼               ▼               ▼
                              MSE > thresh?  logP < thresh?  logP < thresh?
                                    │               │               │
                                    └───────────────┼───────────────┘
                                                    │
                                              ANY True?
                                                    │ Yes
                                                    ▼
                                          Persistence ≥ 5s?
                                                    │ Yes
                                                    ▼
                                                 ALERT
```

### Fallback Handling

If the person is temporarily lost (occlusion, exiting frame):
- The last known good pose is reused for up to 60 frames (~2 seconds)
- After 60 frames without detection, the buffer is cleared
- This prevents false positives from zeroed-out keypoints during brief occlusions

---

## 11. Key Architectural Decisions & Their Justification

| Decision | Justification |
|---|---|
| **Fixed adjacency (not learned)** | Reduces parameters by ~4K; skeleton topology is universal — learning it adds risk of overfitting to camera angle or individual body proportions |
| **Residual connections in every block** | Enables training deeper GCNs without vanishing gradients. Without residuals, 3+ ST-GCN blocks would see degraded gradient flow (He et al., 2016) |
| **AdaptiveAvgPool2d((1,1)) in encoder** | Makes latent dimension independent of $T$ and $V$; allows variable-length sequences without architectural changes |
| **AdaptiveAvgPool2d((30,17)) in decoder** | Fixes the slight dimension mismatch from transposed convolutions (32→30 time steps) with smooth interpolation rather than cropping, which would lose edge frames |
| **Cosine annealing + early stopping** | Prevents overfitting while ensuring convergence. Cosine annealing avoids the sudden LR drops of step-based schedules that can destabilize late-stage training |
| **EMA on MSE ($\alpha=0.3$)** | Reduces jitter without adding detection latency. Lower $\alpha$ = smoother but slower response to genuine changes |
| **5-second persistence filter** | Balances sensitivity vs specificity. 5s is long enough to filter transient false positives, short enough to catch real incidents quickly |
| **Three independent anomaly checks** | Redundancy: covers different failure modes — a fall may have high MSE, unusual crouching may only show in latent GMM, boundary crossing only in spatial GMM |
| **MSE threshold from validation set** | Computing thresholds on held-out data (not training data) gives an honest estimate of what "normal reconstruction error" looks like for unseen normal behavior |
| **Full covariance GMM** | Captures correlations between latent dimensions. Diagonal covariance would miss patterns where two dimensions are jointly unusual even if neither is individually extreme |
| **Scale-position normalization** | Without it, the model would learn camera-specific biases (person always in center, always same distance). Normalization enables generalization to different rooms and setups |

---

## 12. Files & Data Flow

```
collect_stgcn.py                          [Step 1: Data Collection]
  │
  ├──► normal_routine_data_stgcn.npy      Shape: (N, 3, 30, 17)  dtype: float32
  │     └── N = ~2000 windows of normal skeleton movement
  │
  └──► normal_spatial_data.npy            Shape: (M, 3)  dtype: float32
        └── M = number of frames with detected person
        └── Columns: [center_x, center_y, bbox_area]
            │
            ▼
train_stgcn.py                            [Step 2: Training]
  │
  ├──► stgcn_autoencoder.pth              PyTorch checkpoint
  │     ├── model_state_dict              Trained weights (~440K params)
  │     ├── norm_stats                    {mean_x, mean_y, std_x, std_y}
  │     ├── latent_dim: 64
  │     ├── num_joints: 17
  │     ├── time_frames: 30
  │     └── in_features: 3
  │
  ├──► threshold_config_stgcn.json        Detection thresholds
  │     ├── anomaly_threshold             MSE threshold (mean + 3σ)
  │     ├── gmm_anomaly_threshold         Latent GMM threshold (5th percentile)
  │     ├── spatial_gmm_threshold         Spatial GMM threshold (1st percentile)
  │     ├── mean_error, std_error         Reconstruction error stats
  │     └── norm_stats                    Duplicated for standalone loading
  │
  ├──► gmm_scorer.pkl                     3-component sklearn GMM
  │     └── Trained on 64-dim latent vectors
  │
  └──► gmm_spatial.pkl                    5-component sklearn GMM
        └── Trained on 3-dim [cx, cy, area] vectors
            │
            ▼
trial_stgcn.py                            [Step 3: Live Inference]
  │
  ├── Loads stgcn_autoencoder.pth         Model weights
  ├── Loads threshold_config_stgcn.json   All thresholds
  ├── (Optional) Loads gmm_scorer.pkl     Latent anomaly detector
  ├── (Optional) Loads gmm_spatial.pkl    Spatial anomaly detector
  │
  └──► Real-time display
        ├── Green  = Normal
        ├── Orange = Warning (building up, < 5s)
        └── Red    = Confirmed Anomaly (≥ 5s persistence)
```

### Dependency Summary

| Script | Inputs | Outputs | Runtime |
|---|---|---|---|
| `collect_stgcn.py` | Camera, YOLOv8 | 2 `.npy` files | User-controlled (~5-10 min) |
| `train_stgcn.py` | 2 `.npy` files | 1 `.pth`, 1 `.json`, 2 `.pkl` | ~5-15 min (CPU/GPU) |
| `trial_stgcn.py` | Camera, all 4 output files | Live display | Indefinite |

---

## References

1. **ST-GCN** — Yan, S., Xiong, Y., & Lin, D. (2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*. AAAI 2018.
2. **GCN** — Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017.
3. **ResNet** — He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
4. **Batch Normalization** — Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML 2015.
5. **GMM** — Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 9: Mixture Models and EM.
