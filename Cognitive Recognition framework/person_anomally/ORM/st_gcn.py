import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# YOLOv8 Pose (COCO) 17-keypoint Adjacency Matrix
# =============================================================================
# Joint indices:
#   0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
#   5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
#   9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
#   13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
    (0, 5), (0, 6),                           # neck to shoulders
]


def build_adjacency_matrix(num_joints=17, edges=None, strategy='spatial'):
    """
    Build the normalized adjacency matrix for the skeleton graph.

    Args:
        num_joints: number of keypoints (17 for YOLO/COCO)
        edges: list of (src, dst) tuples defining the skeleton
        strategy: 'spatial' for physical connections, 'identity' adds self-loops

    Returns:
        A: (num_joints, num_joints) torch tensor, row-normalized
    """
    if edges is None:
        edges = COCO_SKELETON_EDGES

    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    if strategy in ('spatial', 'both'):
        for src, dst in edges:
            A[src, dst] = 1.0
            A[dst, src] = 1.0  # undirected graph

    # Self-loops (each joint connects to itself)
    for i in range(num_joints):
        A[i, i] = 1.0

    # Symmetric normalization: D^{-1/2} * A * D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=(D > 0), out=np.zeros_like(D, dtype=np.float32))
    D_inv_sqrt[D == 0] = 0.0
    A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

    return torch.tensor(A_norm, dtype=torch.float32)


class STGCNBlock(nn.Module):
    """Single ST-GCN block with residual connection.

    Spatial path:  GCN (einsum with A) → 1×1 Conv → BN → ReLU → Dropout
    Temporal path: Conv2d along time → BN → ReLU → Dropout
    Residual:      1×1 Conv if channel dimensions differ, else identity.
    """
    def __init__(self, in_channels, out_channels, temporal_kernel=3, dropout=0.0):
        super(STGCNBlock, self).__init__()

        # Spatial Graph Convolution
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.spatial_bn = nn.BatchNorm2d(out_channels)

        # Temporal Convolution (slides across the time dimension)
        padding = temporal_kernel // 2
        self.temporal_conv = nn.Conv2d(out_channels, out_channels,
                                       kernel_size=(temporal_kernel, 1),
                                       padding=(padding, 0))
        self.temporal_bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual projection when channel dimensions change
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, A):
        """
        x: (Batch, Channels, Time, Joints)  →  (n, c, t, v)
        A: (Joints, Joints)                  →  (v, w)
        """
        identity = self.residual(x)

        # Spatial message passing via graph adjacency
        x = torch.einsum('nctv,vw->nctw', x, A)
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Temporal motion extraction
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)

        # Residual connection
        x = x + identity
        x = self.relu(x)
        x = self.dropout(x)

        return x


class STGCNEncoder(nn.Module):
    def __init__(self, in_features=3, latent_dim=64):
        super(STGCNEncoder, self).__init__()
        
        # Stack multiple ST-GCN blocks to extract deeper movement patterns
        self.stgcn_layers = nn.ModuleList([
            STGCNBlock(in_features, 16),
            STGCNBlock(16, 32),
            STGCNBlock(32, 64)
        ])
        
        # Pool all time and joint data down into a single mathematical vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x, A):
        # Pass the data and the Adjacency Matrix through the graph layers
        for layer in self.stgcn_layers:
            x = layer(x, A)
            
        # x shape is now: (Batch, 64, 30, 17)
        
        # Compress the tensor into a flat array
        x = self.global_pool(x)  # (Batch, 64, 1, 1)
        x = torch.flatten(x, 1)  # (Batch, 64)
        
        # Final latent vector representing the entire movement window
        latent_vector = self.fc(x) 
        return latent_vector


class STGCNDecoderBlock(nn.Module):
    """Decoder block: upsample spatially, then apply graph convolution to
    maintain skeletal structure. Includes residual connection."""
    def __init__(self, in_channels, out_channels, time_kernel=4, joint_kernel=3,
                 time_stride=2, joint_stride=2, time_pad=1, joint_pad=1,
                 time_out_pad=0, joint_out_pad=0, dropout=0.0):
        super(STGCNDecoderBlock, self).__init__()

        # Transposed convolution for spatial upsampling
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=(time_kernel, joint_kernel),
            stride=(time_stride, joint_stride),
            padding=(time_pad, joint_pad),
            output_padding=(time_out_pad, joint_out_pad)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        x: (B, C_in, T, V)
        returns: (B, C_out, T', V')
        """
        x = self.deconv(x)               # spatial upsampling
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class STGCNDecoder(nn.Module):
    """Graph-aware decoder: upsamples latent vector back to skeleton sequence.

    Uses STGCNDecoderBlock layers that apply graph convolution after each
    upsampling step, so the skeletal structure is preserved during decoding.

    Path: latent(64) → FC → (64,8,5) → (32,16,9) → (16,32,17) → (3,30,17)
    """
    def __init__(self, latent_dim=64, out_features=3, num_joints=17, time_frames=30):
        super(STGCNDecoder, self).__init__()

        self.out_features = out_features
        self.num_joints = num_joints
        self.time_frames = time_frames

        # Initial expansion from latent vector to small spatial map
        self.init_h, self.init_w = 8, 5
        self.fc = nn.Linear(latent_dim, 64 * self.init_h * self.init_w)

        # Stage 1: (64, 8, 5) → (32, 16, 9)
        self.dec1 = STGCNDecoderBlock(64, 32,
            time_kernel=4, joint_kernel=3,
            time_stride=2, joint_stride=2,
            time_pad=1, joint_pad=1)

        # Stage 2: (32, 16, 9) → (16, 32, 17)
        # Joints: (9-1)*2 - 2*1 + 3 = 16-2+3 = 17 ✓ (exact match!)
        self.dec2 = STGCNDecoderBlock(32, 16,
            time_kernel=4, joint_kernel=3,
            time_stride=2, joint_stride=2,
            time_pad=1, joint_pad=1)

        # Final projection to output channels
        self.final_conv = nn.Conv2d(16, out_features, kernel_size=(1, 1))

        # Small adaptive pool: 32 → 30 on time axis only (6% reduction)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((time_frames, num_joints))

    def forward(self, z, A):
        """
        z: (Batch, latent_dim)
        A: (num_joints, num_joints) adjacency matrix (used only for final graph refinement)
        returns: (Batch, out_features, time_frames, num_joints)
        """
        x = self.fc(z)                                    # (B, 64*8*5)
        x = x.view(x.size(0), 64, self.init_h, self.init_w)  # (B, 64, 8, 5)
        x = self.dec1(x)                                  # (B, 32, 16, 9)
        x = self.dec2(x)                                  # (B, 16, 32, 17)
        x = self.final_conv(x)                            # (B, 3, 32, 17)
        # Apply graph convolution at the final 17-joint resolution
        x = torch.einsum('nctv,vw->nctw', x, A)
        x = self.adaptive_pool(x)                         # (B, 3, 30, 17)
        return x


class STGCNAutoencoder(nn.Module):
    """Full ST-GCN Autoencoder for anomaly detection on skeleton sequences.

    Input:  (Batch, Channels=3, Time=30, Joints=17)
    Output: (Batch, Channels=3, Time=30, Joints=17) reconstruction
    """
    def __init__(self, in_features=3, latent_dim=64, num_joints=17, time_frames=30):
        super(STGCNAutoencoder, self).__init__()

        self.num_joints = num_joints
        self.time_frames = time_frames

        # Build the fixed adjacency matrix (not learned by default)
        A = build_adjacency_matrix(num_joints=num_joints)
        self.register_buffer('A', A)  # stored on the correct device, not a learnable parameter

        self.encoder = STGCNEncoder(in_features=in_features, latent_dim=latent_dim)
        self.decoder = STGCNDecoder(latent_dim=latent_dim, out_features=in_features,
                                    num_joints=num_joints, time_frames=time_frames)

    def forward(self, x):
        """
        x: (Batch, 3, 30, 17)
        returns: (Batch, 3, 30, 17) reconstruction, (Batch, latent_dim) latent vector
        """
        latent = self.encoder(x, self.A)
        reconstruction = self.decoder(latent, self.A)
        return reconstruction, latent