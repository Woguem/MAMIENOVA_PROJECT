# ==========================================================
# @author : Dr Yen Fred WOGUEM
# Project: Crystal Plasticity Deep Learning (MAMIE NOVA)
# Description: Prediction of the local geometrically necessary
#              dislocation density field (ρ_GND) with a Fourier Neural Operator (FNO).
# Based on: Li et al. 2021 + Ibragimova et al. 2025
# Framework: PyTorch
# ==========================================================

# --- Standard PyTorch imports ---
import torch, gc
# torch     : main deep-learning framework
# gc        : Python garbage collector, used to free memory manually

import torch.nn as nn
# nn        : module containing all layers (Conv2d, BatchNorm, etc.)

import torch.optim as optim
# optim     : optimisers (Adam, SGD, etc.) — used indirectly via torch.optim.AdamW

from torch.utils.data import Dataset, DataLoader, random_split
# Dataset   : base class for custom datasets
# DataLoader: loads data in batches, in parallel (num_workers)
# random_split : not used directly here, imported for consistency

import numpy as np
# numpy     : numerical array computations (reading .bin files, normalisation)

import pandas as pd
# pandas    : saves the training history to CSV

import seaborn as sns
# seaborn   : statistical visualisation (box plots)

import matplotlib.pyplot as plt
# matplotlib: plots of loss curves and field maps

import os
# os        : paths, directory creation, environment variables

import random
# random    : shuffles the structures for the train/val/test split

from torch.utils.tensorboard import SummaryWriter
# SummaryWriter : writes metrics for visualisation in TensorBoard

import time
# time      : measures the total run time

import torch.nn.functional as F
# F         : parameter-free functions (relu, gelu, etc.)

import json
# json      : saves the train/val/test split for reproducibility


# ==========================================================
# Output channel: ρ_GND only
# ==========================================================
OUTPUT_NAMES = [
    "rhognd"                        # GND dislocation density (m⁻²)
]


# ==========================================================
# Reproducibility: fix all sources of randomness
# ==========================================================
os.environ['PYTHONHASHSEED'] = str(42)
# Forces deterministic Python hashing (otherwise dict/set ordering can change)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# Configures cuBLAS to be deterministic on NVIDIA GPUs
# ':4096:8' = 8 workspace buffers of 4096 KiB

torch.manual_seed(42)
# Sets the PyTorch random seed (CPU)

np.random.seed(42)
# Sets the NumPy seed

random.seed(42)
# Sets the Python random-module seed

torch.backends.cudnn.deterministic = True
# Forces cuDNN to use deterministic algorithms (slower but reproducible)

torch.backends.cudnn.benchmark = False
# Disables cuDNN auto-tuning (non-deterministic)

torch.use_deterministic_algorithms(False)
# Raises an error if a non-deterministic PyTorch operation is called

start_time = time.time()
# Records the start time to compute the total run time at the end


# ==========================================================
# 1. Custom dataset for CPFFT binary files
# ==========================================================
class CrystalPlasticityDataset(Dataset):
    """
    Reads the binary outputs of CPFFT simulations.
    Each structure_XXX folder contains:
      - Outputs/Outputs_0_YYY.bin : local fields at each strain step
      - Outputs/rho0.txt          : initial dislocation density
    """

    def __init__(self, base_dir, input_ranges, output_ranges,
                 file_list=None, augment=False):
        super().__init__()
        self.base_dir      = base_dir
        # Root directory containing all structure_* folders

        self.input_ranges  = input_ranges
        # Dict with the input normalisation ranges
        # {"epsilon": (min, max), "rho0": (min, max)}

        self.output_ranges = output_ranges
        # Dict with the output normalisation range
        # {"rhognd": (min, max)}

        self.epsilon_range = input_ranges["epsilon"]
        # Shortcut: range of the applied macroscopic strain

        self.rho0_range    = input_ranges["rho0"]
        # Shortcut: range of the initial dislocation density

        self.augment  = augment
        # If True: returns 4 versions per sample (original + 3 flips/rotations)
        # Used for the training set only

        self.file_list = file_list
        # Explicit list of .bin files (None = use the generator)

        # Total dataset length
        self._length = (
            sum(1 for _ in self._load_all_file_paths())
            if file_list is None
            else len(file_list)
        )

    def normalize(self, x, r):
        """Normalises x to [-1, 1] using the range r=(min, max)."""
        denom = r[1] - r[0]
        if denom == 0:
            raise ValueError(f"Invalid range (min=max): {r}")
        return 2.0 * (x - r[0]) / denom - 1.0

    def rotate_180_x(self, x, y):
        """Vertical flip (top ↔ bottom) for data augmentation."""
        return np.flip(x, axis=1).copy(), np.flip(y, axis=1).copy()

    def rotate_180_y(self, x, y):
        """Horizontal flip (left ↔ right) for data augmentation."""
        return np.flip(x, axis=2).copy(), np.flip(y, axis=2).copy()

    def _load_all_file_paths(self):
        """
        Generator: walks through all structure_* folders and
        yields the path of every valid .bin file
        (excluding Outputs_0_000.bin, the initial step ε=0).
        """
        sim_dirs = sorted([
            d for d in os.listdir(self.base_dir)
            if d.startswith("structure_")
        ])
        # Alphabetical sort for a deterministic order

        for sim_dir_name in sim_dirs:
            sim_dir    = os.path.join(self.base_dir, sim_dir_name)
            outputs_dir = os.path.join(sim_dir, "Outputs")

            if not os.path.exists(outputs_dir):
                raise FileNotFoundError(
                    f"Missing Outputs folder: {sim_dir}")

            bin_files = [
                os.path.join(outputs_dir, f)
                for f in os.listdir(outputs_dir)
                if f.startswith("Outputs_0_")
                and f.endswith(".bin")
                and f != "Outputs_0_000.bin"
                # Excludes step zero: all fields are zero, not informative
            ]
            if not bin_files:
                raise FileNotFoundError(
                    f"No valid .bin file in: {outputs_dir}")

            for bin_file in sorted(bin_files):
                # sorted() keeps the strain steps in chronological order
                yield bin_file

    def get_structure_name(self, bin_file):
        """Extracts the structure name from a .bin file path."""
        for part in bin_file.split(os.sep):
            if part.startswith("structure_"):
                return part
        raise ValueError(f"Cannot extract the structure name from: {bin_file}")

    def filter_files_by_structures(self, structures):
        """Returns the list of .bin files belonging to the requested structures."""
        return [
            bf for bf in self._load_all_file_paths()
            if self.get_structure_name(bf) in structures
        ]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """
        Returns one sample (or 4 if augment=True).
        Each sample is a tuple (X, Y, bin_file) with:
          X : tensor (6, 128, 128) — normalised inputs
          Y : tensor (1, 128, 128) — normalised ρ_GND
        """
        bin_file = (
            self.file_list[idx]
            if self.file_list is not None
            else next(
                x for i, x in enumerate(self._load_all_file_paths())
                if i == idx
            )
        )
        # If file_list is given: direct access by index (O(1))
        # Otherwise: advances the generator up to idx (O(n)) — avoid in production

        x, y = self.load_data_pair(bin_file)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if self.augment:
            # Generates 3 additional transformed versions
            x_rot_x,  y_rot_x  = self.rotate_180_x(x, y)
            x_rot_y,  y_rot_y  = self.rotate_180_y(x, y)
            x_rot_yx, y_rot_yx = self.rotate_180_x(x_rot_y, y_rot_y)
            # rot_yx = flip in y THEN in x = 180° in-plane rotation

            return [
                (torch.from_numpy(x),        torch.from_numpy(y),        bin_file),
                (torch.from_numpy(x_rot_x),  torch.from_numpy(y_rot_x),  bin_file),
                (torch.from_numpy(x_rot_y),  torch.from_numpy(y_rot_y),  bin_file),
                (torch.from_numpy(x_rot_yx), torch.from_numpy(y_rot_yx), bin_file),
            ]
            # Returns a list of 4 tuples → stacked by collate_fn_train

        return torch.from_numpy(x), torch.from_numpy(y), bin_file

    def read_fortran_record(self, f, shape, dtype=np.float32):
        """
        Reads one Fortran unformatted binary record.
        Format: [4-byte marker] [data] [4-byte marker]
        The 4-byte markers store the record size.
        """
        f.read(4)   # Skips the leading marker
        data = np.fromfile(f, dtype=dtype, count=np.prod(shape))
        f.read(4)   # Skips the trailing marker
        return data.reshape(shape, order='F')
        # order='F' = Fortran (column-major) order vs C (row-major) order

    def euler_to_quat(self, phi1, Phi, phi2):
        """
        Converts Euler angles (φ₁, Φ, φ₂) to unit quaternions.
        Quaternions represent the crystallographic orientation of each grain.
        They are numerically more stable than Euler angles (no singularity).
        Returns: array (4, H, W) with q1, q2, q3, q4
        """
        Ang_plus  = (phi1 + phi2) / 2   # mean angle
        Ang_moins = (phi2 - phi1) / 2   # half-difference
        sbeta     = Phi / 2              # half-tilt

        q1 = np.cos(sbeta) * np.cos(Ang_plus)
        q2 = np.sin(sbeta) * np.cos(Ang_moins)
        q3 = -np.sin(sbeta) * np.sin(Ang_moins)
        q4 = np.cos(sbeta) * np.sin(Ang_plus)

        norm = np.sqrt(q1**2 + q2**2 + q3**2 + q4**2)
        quaternions = np.stack([q1, q2, q3, q4], axis=0) / norm
        # Normalisation to guarantee ||q|| = 1

        quaternions[:, q4 < 0] *= -1
        # Convention: q4 always positive (single hemisphere)
        # Prevents two identical orientations from having opposite quaternions

        return quaternions

    def load_data_pair(self, bin_file):
        """
        Loads and prepares one (X, Y) pair from a .bin file.
        X shape : (6, 128, 128)  → [ε_macro, ρ₀, q1, q2, q3, q4]
        Y shape : (1, 128, 128)  → [ρGND]
        """
        # ── Extract ε_macro from the file name ───────────────────────────
        file_name = os.path.basename(bin_file)
        # Example: Outputs_0_025.bin → eps_macro = 0.025
        eps_macro_str = file_name.split('_')[2].split('.')[0]
        epsilon_macro_11 = float("0." + eps_macro_str)
        # Parse "025" → "0.025" → 0.025

        # ── Read ρ₀ from rho0.txt ────────────────────────────────────────
        rho0_file = os.path.join(os.path.dirname(bin_file), 'rho0.txt')
        if not os.path.isfile(rho0_file):
            raise FileNotFoundError(
                f"Missing rho0.txt file in {os.path.dirname(bin_file)}")
        with open(rho0_file, 'r') as f:
            text = f.read().replace('d', 'e').replace('D', 'e').strip()
            # 'd' and 'D' are Fortran exponents (e.g. 1.0d10 → 1.0e10)
        rho0 = float(text)

        # ── Read the binary file ─────────────────────────────────────────
        nx, ny, nz = 128, 128, 1
        # Spatial grid: 128×128 pixels, 1 layer (2D problem)

        # All records are read in file order to reach the required fields;
        # only ρ_GND and the Euler angles are used.
        with open(bin_file, 'rb') as f:
            eps11  = self.read_fortran_record(f, (nx, ny, nz))  # local strain ε₁₁
            eps22  = self.read_fortran_record(f, (nx, ny, nz))  # local strain ε₂₂
            eps33  = self.read_fortran_record(f, (nx, ny, nz))  # local strain ε₃₃
            _      = self.read_fortran_record(f, (nx, ny, nz))  # ε₃₂ (unused)
            _      = self.read_fortran_record(f, (nx, ny, nz))  # ε₃₁ (unused)
            eps12  = self.read_fortran_record(f, (nx, ny, nz))  # local strain ε₁₂
            sig11  = self.read_fortran_record(f, (nx, ny, nz))  # local stress σ₁₁
            sig22  = self.read_fortran_record(f, (nx, ny, nz))  # local stress σ₂₂
            _      = self.read_fortran_record(f, (nx, ny, nz))  # σ₃₃ (unused)
            _      = self.read_fortran_record(f, (nx, ny, nz))  # σ₃₂ (unused)
            _      = self.read_fortran_record(f, (nx, ny, nz))  # σ₃₁ (unused)
            sig12  = self.read_fortran_record(f, (nx, ny, nz))  # local stress σ₁₂
            rhognd = self.read_fortran_record(f, (nx, ny, nz))  # GND dislocation density
            rhossd = self.read_fortran_record(f, (nx, ny, nz))  # SSD dislocation density
            phi1   = self.read_fortran_record(f, (nx, ny, nz))  # Euler angle φ₁
            phi    = self.read_fortran_record(f, (nx, ny, nz))  # Euler angle Φ
            phi2   = self.read_fortran_record(f, (nx, ny, nz))  # Euler angle φ₂

        H, W = ny, nx  # H=128, W=128

        # ── Build the inputs X ───────────────────────────────────────────

        # Euler → quaternion conversion (uses the z=0 slice of the 3D volume)
        quaternions = self.euler_to_quat(
            phi1[:, :, 0], phi[:, :, 0], phi2[:, :, 0])
        # Resulting shape: (4, 128, 128)

        # Normalisation of ε_macro to [-1, 1]
        epsilon_norm = (
            2 * (epsilon_macro_11 - self.epsilon_range[0])
            / (self.epsilon_range[1] - self.epsilon_range[0]) - 1
        )
        epsilon_M = np.full((1, H, W), epsilon_norm, dtype=np.float32)
        # Broadcast: same scalar value over the whole 128×128 map
        # This tells the model which strain level it is working at

        # Normalisation of ρ₀ on a log scale, then to [-1, 1]
        rho0_log     = np.log10(rho0 + 1.0)
        # Log because ρ₀ spans 4 decades (1e10 to 1e14) → more uniform log scale
        # +1 to avoid log(0)
        rho0_log_min = np.log10(self.rho0_range[0] + 1.0)
        rho0_log_max = np.log10(self.rho0_range[1] + 1.0)
        rho0_norm    = 2 * (rho0_log - rho0_log_min) / (rho0_log_max - rho0_log_min) - 1
        rho0_M       = np.full((1, H, W), rho0_norm, dtype=np.float32)

        # Final assembly of the 6 input channels
        X = np.concatenate([
            epsilon_M,   # channel 0: applied macroscopic strain
            rho0_M,      # channel 1: initial dislocation density
            quaternions  # channels 2-5: local crystallographic orientation
        ], axis=0)       # Shape: (6, 128, 128)

        # ── Build the output Y ───────────────────────────────────────────

        # ρ_GND: log before normalisation (same reason as ρ₀)
        rhognd_log  = np.log10(rhognd + 1.0)
        rhognd_norm = self.normalize(rhognd_log, self.output_ranges["rhognd"])

        # Single-channel output (z=0 slice → 2D)
        Y = rhognd_norm[np.newaxis, :, :, 0]  # Shape: (1, 128, 128)

        return X.astype(np.float32), Y.astype(np.float32)


def collate_fn_train(batch):
    """
    Collate function for the DataLoader in training mode (augment=True).
    Each batch item is a list of 4 tuples → they are unpacked and stacked.
    Result: effective batch size of 4 × batch_size.
    """
    all_samples = []
    for item in batch:
        all_samples.extend(item)  # Unpacks the list of 4 tuples

    x_batch = torch.stack([s[0] for s in all_samples])  # (4*B, 6, H, W)
    y_batch = torch.stack([s[1] for s in all_samples])  # (4*B, 1, H, W)
    files   = [s[2] for s in all_samples]
    return x_batch, y_batch, files


def collate_fn_val(batch):
    """
    Collate function for validation/test (no augmentation).
    Each item is a single tuple → stacked directly.
    """
    x_batch = torch.stack([s[0] for s in batch])  # (B, 6, H, W)
    y_batch = torch.stack([s[1] for s in batch])  # (B, 1, H, W)
    files   = [s[2] for s in batch]
    return x_batch, y_batch, files


# ==========================================================
# 2. FNO architecture (Fourier Neural Operator)
# ==========================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # ✅ FIX: forces a 4D shape regardless of DataParallel replication
        w1 = self.weights1.view(self.in_channels, self.out_channels,
                                self.modes1, self.modes2)
        w2 = self.weights2.view(self.in_channels, self.out_channels,
                                self.modes1, self.modes2)

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2], w1)

        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2], w2)

        return torch.fft.irfft2(out_ft, s=(H, W))

class FNOBlock2d(nn.Module):
    """
    Complete FNO block = spectral path + 1×1 bypass, with a residual connection.

    Parallel architecture:
      ┌─ SpectralConv2d ──────────┐
    x ┤                           ├─→ Norm → Act → output
      └─ Conv2d(1×1) [bypass] ───┘

    The bypass (1×1 conv):
      - Captures local information (smoothed by the FFT)
      - Keeps the gradient flowing even if the spectral branch saturates
      - Makes the block equivalent to a ResNet in spectral space
    """

    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        # Global branch: convolution in Fourier space

        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        # Bypass branch: 1×1 convolution = per-pixel linear transformation
        # No padding_mode="circular" needed: a 1×1 kernel does not slide

        self.norm = nn.InstanceNorm2d(channels, affine=True)
        # InstanceNorm (not BatchNorm) because:
        #   - Physical fields have very different statistics across samples
        #   - InstanceNorm normalises each channel of each sample independently
        #   - affine=True: learns rescaling parameters γ and β

        self.act = nn.GELU()
        # GELU (Gaussian Error Linear Unit): smooth activation, better than ReLU
        # for transformers and spectral operators

    def forward(self, x):
        residual = x  # Saved for the skip connection
        x = self.act(self.norm(self.spectral(x) + self.bypass(x)))
        return x + residual
        # Residual connection around (spectral + bypass) → Norm → Act


class FNOModel2d(nn.Module):
    """
    Complete 2D Fourier Neural Operator for MAMIE NOVA.

    Data flow:
      X (6ch) → [+coordinate grid] → Lifting (width ch)
              → N × FNOBlock2d
              → Projection → Y (1ch: ρ_GND) ∈ [-1,1]

    Parameters:
      in_channels  : 6 (ε_macro, ρ₀, q1, q2, q3, q4)
      out_channels : 1 (ρGND)
      modes        : retained Fourier modes per direction
      width        : number of internal (latent) channels
      n_layers     : number of FNO blocks
    """

    def __init__(self, in_channels=6, out_channels=1,
                 modes=24, width=64, n_layers=4):
        super().__init__()
        self.modes   = modes
        self.width   = width

        # ── Lifting: projects the input to the latent space ──────────────
        self.lifting = nn.Conv2d(in_channels + 2, width, kernel_size=1)
        # +2 because the normalised spatial coordinates (x, y) are concatenated
        # 1×1 conv = independent per-pixel linear projection
        # width channels → sufficiently rich latent space

        # ── FNO blocks ───────────────────────────────────────────────────
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(width, modes, modes)
            for _ in range(n_layers)
        ])
        # ModuleList makes PyTorch register the parameters correctly
        # All blocks have the same dimension (width→width)

        # ── Projection: latent space → output channel ────────────────────
        self.projection = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            # Increases capacity before the final projection (inverted bottleneck)
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
            # Reduces to the output channel (ρ_GND)
            nn.Tanh()
            # Tanh constrains the output to [-1, 1], consistent with the normalisation
        )

    def get_grid(self, shape, device):
        """
        Generates a grid of normalised coordinates in [0, 1]×[0, 1].
        Needed because the FNO does not "know" where it is in space
        (Fourier modes are translation-invariant).
        The grid breaks this invariance and helps localise structures.
        """
        B, _, H, W = shape
        grid_h = (torch.linspace(0, 1, H, device=device)
                  .view(1, 1, H, 1).expand(B, 1, H, W))
        # y coordinate: from 0 (top) to 1 (bottom), broadcast over batch and W
        grid_w = (torch.linspace(0, 1, W, device=device)
                  .view(1, 1, 1, W).expand(B, 1, H, W))
        # x coordinate: from 0 (left) to 1 (right), broadcast over batch and H
        return torch.cat([grid_h, grid_w], dim=1)  # (B, 2, H, W)

    def forward(self, x):
        # Concatenates the spatial coordinates to the input features
        grid = self.get_grid(x.shape, x.device)  # (B, 2, H, W)
        x = torch.cat([x, grid], dim=1)           # (B, 8, H, W)

        # Lifting to the latent space
        x = self.lifting(x)   # (B, width, H, W)

        # Successive FNO blocks
        for block in self.fno_blocks:
            x = block(x)      # (B, width, H, W) — dimension preserved

        # Projection to the output channel (ρ_GND)
        return self.projection(x)  # (B, 1, H, W) ∈ [-1, 1]


def MamieNovaFNOModel(in_channels=6, out_channels=1,
                      modes=24, width=64, n_layers=4):
    """Main entry point — replaces MamieNovaUNETModel."""
    return FNOModel2d(in_channels, out_channels, modes, width, n_layers)


# ==========================================================
# 3. Training function
# ==========================================================

def train_model(model, train_loader, val_loader, device,
                save_dir, epochs, lr, output_ranges, modes=None, width=None, n_layers=None, history_suffix=''):

    criterion = nn.MSELoss(reduction="mean")
    # MSE (Mean Squared Error) = penalises large errors more strongly
    # Suitable because the fields are normalised to [-1, 1]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        # beta1=0.9 : momentum on the gradient (inertia)
        # beta2=0.999 : momentum on the squared gradient (step-size adaptation)
        eps=1e-8,
        # Avoids division by zero in the weight update
        weight_decay=1e-4
        # Decoupled weight decay: penalises large weights, reduces overfitting
    )

    # Cosine annealing: decreases the lr along a cosine curve
    # Helps the model settle in a flatter minimum (better generalisation)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer,
    #    T_max=epochs,     # full period = total number of epochs
    #    eta_min=1e-6      # minimum lr reached at the end
    #)

    history = {"epoch": [], "train_loss": [], "val_loss": []}

    patience = 15
    # Early stopping: stops after 15 evaluations without improvement
    # (15 × 3 epochs = 45 epochs without improvement → stop)
    # Raised to 15 because the FNO converges more slowly than the U-Net

    best_val_loss          = float('inf')
    epochs_without_improvement = 0
    best_model_path        = os.path.join(save_dir, "best_model.pth")
    history_csv_path       = os.path.join(save_dir, f"training_history{history_suffix}.csv")

    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_logs'))
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):

        # ── Training phase ───────────────────────────────────────────────
        model.train()
        # Enables training-specific layer behaviour: BatchNorm (running stats),
        # Dropout (random masking), etc.

        train_loss = 0.0

        for x_batch, y_batch, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Moves the tensors to the GPU (asynchronous with pin_memory=True)

            optimizer.zero_grad()
            # Resets the gradients BEFORE the forward pass
            # (PyTorch accumulates gradients by default)

            y_pred = model(x_batch)
            # Forward pass: computes the prediction (B, 1, H, W)

            loss = criterion(y_pred, y_batch)
            # MSE on the ρ_GND channel normalised to [-1, 1]

            loss.backward()
            # Backward pass: computes ∂loss/∂(all parameters) by the chain rule

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Gradient clipping: caps the gradient norm at 1.0
            # CRUCIAL for the FNO: FFT operations can amplify gradients
            # Without clipping → numerical instability / exploding gradients

            optimizer.step()
            # Updates the weights: θ ← θ - lr × Adam(∇loss)

            # Accumulates the loss (weighted by the batch size)
            n = x_batch.size(0)
            train_loss     += loss.item()     * n

        # Normalises by the total number of samples (×4 due to augmentation)
        denom = 4 * len(train_loader.dataset)
        train_loss     /= denom

        #scheduler.step()
        # Updates the learning rate along the cosine schedule

        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        #writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch + 1)
        # Also logs the lr to check that the schedule works

        # ── Validation phase (every 3 epochs) ────────────────────────────
        val_loss = None

        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            model.eval()
            # Disables BatchNorm updates (uses global stats) and Dropout

            val_loss = 0.0

            with torch.no_grad():
                # Disables gradient computation → less GPU memory,
                # much faster validation

                for x_val, y_val, _ in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_pred = model(x_val)

                    loss = criterion(y_pred, y_val)

                    n = x_val.size(0)
                    val_loss     += loss.item()     * n

            denom_val = len(val_loader.dataset)
            val_loss     /= denom_val

            writer.add_scalar('Loss/val', val_loss, epoch + 1)

            # Saves the best model
            if val_loss < best_val_loss:
                best_val_loss              = val_loss
                epochs_without_improvement = 0

                # Extracts the state_dict without the DataParallel "module." prefix
                state_dict_to_save = (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                )
                torch.save({
                    'epoch':      epoch + 1,
                    'state_dict': state_dict_to_save,
                    'val_loss':   best_val_loss,
                    'model_cfg':  {'modes': modes, 'width': width, 'n_layers': n_layers}
                    # Saves the config so that the model can be rebuilt later
                }, best_model_path)

                print(f"✅ Best model saved at epoch {epoch+1}, "
                      f"val_loss={val_loss:.4f}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

        # Frees GPU memory at each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Records the history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Saves the CSV every 3 epochs
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            history_df = pd.DataFrame(history)
            history_df.to_csv(history_csv_path, index=False)
            print(f"✅ History saved → {history_csv_path}")

        if val_loss is not None:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f}")

    # Reloads the best weights at the end
    if os.path.exists(best_model_path):
        checkpoint  = torch.load(best_model_path, map_location=device,
                                 weights_only=False)
        state_dict  = {k.replace("module.", ""): v
                       for k, v in checkpoint["state_dict"].items()}
        target = (model.module if isinstance(model, nn.DataParallel) else model)
        target.load_state_dict(state_dict, strict=True)
        model.to(device)
        print(f"✅ Best weights reloaded (epoch {checkpoint['epoch']}, "
              f"val_loss={checkpoint['val_loss']:.4f})")

    # Final loss curve
    training_curves_path = os.path.join(save_dir, "training_curves.png")
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=history_df, x="epoch", y="train_loss", label="Train Loss")
    sns.lineplot(data=history_df, x="epoch", y="val_loss",   label="Val Loss")
    plt.title("Training & Validation Loss (FNO)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(training_curves_path)
    plt.close()
    print(f"✅ Curve saved → {training_curves_path}")

    writer.close()
    return model, history_df


# ==========================================================
# 4. Test function
# ==========================================================

def count_outliers(data):
    """Counts the IQR outliers in a distribution."""
    if len(data) == 0:
        return 0, 0, 0, 0
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = [x for x in data if x < lb or x > ub]
    pct = len(outliers) / len(data) * 100
    return len(outliers), pct, lb, ub


def test_model(model, test_loader, device, save_dir,
               input_ranges, output_ranges, full_dataset, test_set):

    model.to(device)
    model.eval()
    output_names = OUTPUT_NAMES

    rmse_errors    = {name: [] for name in output_names}
    mae_errors     = {name: [] for name in output_names}
    structure_data = {}
    sample_idx     = 0
    # sample_idx tracks the global position in test_set.file_list
    # (needed to recover the exact file path of each sample)

    with torch.no_grad():
        for x_test, y_test, _ in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_pred = model(x_test)

            y_true = y_test.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            # ── Denormalisation ──────────────────────────────────────────
            y_true_denorm = np.zeros_like(y_true)
            y_pred_denorm = np.zeros_like(y_pred)

            for i, name in enumerate(output_names):
                r = output_ranges[name]
                # Inverse of the normalisation: x = val * (max-min)/2 + (max+min)/2
                if name in ["rhossd", "rhognd"]:
                    # These channels were log-transformed before normalisation
                    # → denormalise first, then 10^x - 1 to recover m⁻²
                    y_true_denorm[:, i] = (
                        10 ** (y_true[:, i] * (r[1]-r[0])/2 + (r[0]+r[1])/2) - 1)
                    y_pred_denorm[:, i] = (
                        10 ** (y_pred[:, i] * (r[1]-r[0])/2 + (r[0]+r[1])/2) - 1)
                else:
                    y_true_denorm[:, i] = (
                        y_true[:, i] * (r[1]-r[0])/2 + (r[0]+r[1])/2)
                    y_pred_denorm[:, i] = (
                        y_pred[:, i] * (r[1]-r[0])/2 + (r[0]+r[1])/2)

            # ── RMSE / MAE per map ───────────────────────────────────────
            for i, name in enumerate(output_names):
                for j in range(y_true.shape[0]):
                    t = y_true_denorm[j, i]
                    p = y_pred_denorm[j, i]
                    rmse_errors[name].append(np.sqrt(np.mean((t - p)**2)))
                    mae_errors[name].append(np.mean(np.abs(t - p)))
                    # RMSE: sensitive to large localised errors
                    # MAE : more robust to outliers

            # ── Per-structure storage ────────────────────────────────────
            for j in range(y_true.shape[0]):
                bin_file       = test_set.file_list[sample_idx]
                structure_name = full_dataset.get_structure_name(bin_file)
                sample_idx    += 1

                if structure_name not in structure_data:
                    structure_data[structure_name] = {
                        "Strain":     [],
                        "Quaternion": [],
                        "True":       {name: [] for name in output_names},
                        "Predicted":  {name: [] for name in output_names}
                    }

                # Recovers the scalar strain value of this sample
                strain_val = (
                    x_test[j, 0, 0, 0].cpu().numpy()
                    * (input_ranges["epsilon"][1] - input_ranges["epsilon"][0]) / 2
                    + (input_ranges["epsilon"][0] + input_ranges["epsilon"][1]) / 2
                )
                structure_data[structure_name]["Strain"].append(strain_val)
                structure_data[structure_name]["Quaternion"].append(
                    x_test[j, 2:6].cpu().numpy())

                for i, name in enumerate(output_names):
                    structure_data[structure_name]["True"][name].append(
                        y_true_denorm[j, i])
                    structure_data[structure_name]["Predicted"][name].append(
                        y_pred_denorm[j, i])

    # ── Save one .npz per structure ──────────────────────────────────────
    for structure_name, data in structure_data.items():
        save_dict = {
            "strain":     np.array(data["Strain"]),
            "quaternion": np.stack(data["Quaternion"], axis=0),
        }
        for name in output_names:
            save_dict[f"true_{name}"] = np.stack(data["True"][name],      axis=0)
            save_dict[f"pred_{name}"] = np.stack(data["Predicted"][name], axis=0)

        save_path = os.path.join(save_dir, f"{structure_name}.npz")
        np.savez(save_path, **save_dict)
        print(f"✅ Saved: {save_path}")

    # ── Text report ──────────────────────────────────────────────────────
    output_file = os.path.join(save_dir, "test_results_info.txt")
    with open(output_file, 'w') as f:
        f.write("=== SUMMARY STATISTICS (FNO) ===\n\n")
        for name in output_names:
            rd, md = rmse_errors[name], mae_errors[name]
            if rd:
                rq1, rq3 = np.percentile(rd, [25, 75])
                mq1, mq3 = np.percentile(md, [25, 75])
                f.write(f"--- {name} ---\n")
                f.write(f"RMSE : Mean={np.mean(rd):.4f}, Q1={rq1:.4f}, Q3={rq3:.4f}\n")
                f.write(f"MAE  : Mean={np.mean(md):.4f}, Q1={mq1:.4f}, Q3={mq3:.4f}\n\n")

        f.write("\n=== OUTLIERS RMSE ===\n")
        for name in output_names:
            rv = rmse_errors[name]
            if rv:
                n, pct, lb, ub = count_outliers(rv)
                f.write(f"{name}: N={len(rv)}, outliers={n} ({pct:.1f}%), "
                        f"LB={lb:.4f}, UB={ub:.4f}\n")

        f.write("\n=== OUTLIERS MAE ===\n")
        for name in output_names:
            mv = mae_errors[name]
            if mv:
                n, pct, lb, ub = count_outliers(mv)
                f.write(f"{name}: N={len(mv)}, outliers={n} ({pct:.1f}%), "
                        f"LB={lb:.4f}, UB={ub:.4f}\n")

    print(f"✅ Report → {output_file}")

    # ── RMSE / MAE box plots ─────────────────────────────────────────────
    for metric_name, errors in [("rmse", rmse_errors), ("mae", mae_errors)]:
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=[errors[n] for n in output_names])
        plt.xticks(range(len(output_names)), output_names, rotation=45)
        plt.ylabel(metric_name.upper())
        plt.title(f"Distribution of {metric_name.upper()} errors per channel (FNO)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric_name}_boxplots_all_channels.png"))
        plt.close()

    print(f"✅ Box plots saved in {save_dir}")


# ==========================================================
# Utility: computing the normalisation ranges
# ==========================================================

def read_fortran_record(f, shape, dtype=np.float32):
    """Reads one Fortran binary record (standalone version)."""
    f.read(4)
    data = np.fromfile(f, dtype=dtype, count=np.prod(shape))
    f.read(4)
    return data.reshape(shape, order='F')


def calculate_output_ranges(base_dir):
    """
    Walks through the WHOLE dataset to find the global min/max of log10(ρ_GND + 1).
    This range is used to normalise the target to [-1, 1].
    Note: computed over train+val+test (a single range shared by all
    splits).
    """
    output_ranges = {
        name: {"min": float('inf'), "max": float('-inf')}
        for name in OUTPUT_NAMES
    }
    sim_dirs = sorted([
        d for d in os.listdir(base_dir) if d.startswith("structure_")
    ])

    for sim in sim_dirs:
        outputs_dir = os.path.join(base_dir, sim, "Outputs")
        bin_files   = sorted(
            f for f in os.listdir(outputs_dir)
            if f.startswith("Outputs_0_") and f.endswith(".bin")
        )
        for bf in bin_files:
            with open(os.path.join(outputs_dir, bf), "rb") as f:
                nx, ny, nz = 128, 128, 1
                eps11  = read_fortran_record(f, (nx, ny, nz))
                eps22  = read_fortran_record(f, (nx, ny, nz))
                eps33  = read_fortran_record(f, (nx, ny, nz))
                _      = read_fortran_record(f, (nx, ny, nz))  # eps32
                _      = read_fortran_record(f, (nx, ny, nz))  # eps31
                eps12  = read_fortran_record(f, (nx, ny, nz))
                sig11  = read_fortran_record(f, (nx, ny, nz))
                sig22  = read_fortran_record(f, (nx, ny, nz))
                _      = read_fortran_record(f, (nx, ny, nz))  # sig33
                _      = read_fortran_record(f, (nx, ny, nz))  # sig32
                _      = read_fortran_record(f, (nx, ny, nz))  # sig31
                sig12  = read_fortran_record(f, (nx, ny, nz))
                rhognd = read_fortran_record(f, (nx, ny, nz))
                rhossd = read_fortran_record(f, (nx, ny, nz))

                # Log transform of ρ_GND (same processing as in load_data_pair)
                rhognd_log = np.log10(rhognd + 1.0)

                for name, arr in [("rhognd", rhognd_log)]:
                    output_ranges[name]["min"] = min(
                        output_ranges[name]["min"], float(np.min(arr)))
                    output_ranges[name]["max"] = max(
                        output_ranges[name]["max"], float(np.max(arr)))

    return {
        name: (vals["min"], vals["max"])
        for name, vals in output_ranges.items()
    }


# ==========================================================
# 5. Main
# ==========================================================
if __name__ == "__main__":

    # ── Paths from environment variables ─────────────────────────────────
    base_dir = os.getenv('DATASET_PATH')
    save_dir = os.getenv('RESULTS_PATH')

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"❌ Dataset not found: {base_dir}")
    os.makedirs(save_dir, exist_ok=True)
    if not os.access(save_dir, os.W_OK):
        raise PermissionError(f"❌ Cannot write to: {save_dir}")

    print(f"📁 Dataset : {base_dir}")
    print(f"💾 Results : {save_dir}")

    # ── Normalisation ranges ─────────────────────────────────────────────
    input_ranges = {
        "epsilon": (0.0, 0.05),   # macroscopic strain: 0% to 5%
        "rho0":    (1e10, 1e14)   # initial density: 4 decades
    }
    print("🔢 Computing the ρ_GND normalisation range...")
    output_ranges = calculate_output_ranges(base_dir)
    print(f"Output ranges : {output_ranges}")

    # ── Configuration ────────────────────────────────────────────────────
    config = {
        'batch_size':    12,
        'epochs':        100,
        'learning_rate': 1e-4,
        'train_ratio':   0.8,
        'val_ratio':     0.1,
        'test_ratio':    0.1,
        # FNO config
        'fno_modes':     16,
        # 16 retained modes per direction on a 128×128 grid
        'fno_width':     64,
        # 64 internal channels = good accuracy/memory trade-off
        'fno_layers':    4,
        # Number of blocks (global receptive field from the 1st block thanks to the FFT)
    }

    # ── Full dataset ─────────────────────────────────────────────────────
    full_dataset = CrystalPlasticityDataset(
        base_dir, input_ranges=input_ranges, output_ranges=output_ranges)

    if len(full_dataset) == 0:
        raise ValueError("❌ Empty dataset. Check the paths.")
    print(f"📁 {len(full_dataset)} files loaded (textured + non-textured)")

    # ── Train / val / test split by structure ────────────────────────────
    sim_dirs = sorted([
        d for d in os.listdir(base_dir) if d.startswith("structure_")
    ])
    random.shuffle(sim_dirs)
    # random.seed(42) set above → identical shuffle at every run

    n_total = len(sim_dirs)
    n_train = int(config['train_ratio'] * n_total)
    n_val   = int(config['val_ratio']   * n_total)

    train_structures = sim_dirs[:n_train]
    val_structures   = sim_dirs[n_train:n_train + n_val]
    test_structures  = sim_dirs[n_train + n_val:]

    # ⭐ Saves the split for future reproducibility
    split_path = os.path.join(save_dir, "data_split.json")
    with open(split_path, 'w') as f:
        json.dump({
            "train": train_structures,
            "val":   val_structures,
            "test":  test_structures
        }, f, indent=2)
    print(f"✅ Split saved → {split_path}")
    # Allows reloading exactly the same structures without relying on the seed

    # Filters the files of each set
    train_files = full_dataset.filter_files_by_structures(train_structures)
    val_files   = full_dataset.filter_files_by_structures(val_structures)
    test_files  = full_dataset.filter_files_by_structures(test_structures)

    # Creates the final datasets
    train_set = CrystalPlasticityDataset(
        base_dir, input_ranges, output_ranges,
        file_list=train_files, augment=True)
    # augment=True: ×4 samples by flips/rotation → better invariance to symmetries

    val_set = CrystalPlasticityDataset(
        base_dir, input_ranges, output_ranges,
        file_list=val_files, augment=False)

    test_set = CrystalPlasticityDataset(
        base_dir, input_ranges, output_ranges,
        file_list=test_files, augment=False)

    # ── DataLoaders ──────────────────────────────────────────────────────
    default_num_workers = 12
    # Parallel workers for loading data from disk
    # Recommended: ~2 × number of available CPU cores

    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'],
        shuffle=True,                    # Random shuffle at each epoch
        collate_fn=collate_fn_train,     # Unpacks the 4 augmentations
        num_workers=default_num_workers,
        pin_memory=True                  # Pinned memory → faster GPU transfer
    )
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'],
        shuffle=False,                   # No shuffle for validation
        collate_fn=collate_fn_val,
        num_workers=default_num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=1,
        shuffle=False,                   # MANDATORY: sample_idx depends on the order
        collate_fn=collate_fn_val,
        num_workers=default_num_workers,
        pin_memory=True
    )

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device : {device}")

    # ── Build the FNO model ──────────────────────────────────────────────
    model = MamieNovaFNOModel(
        in_channels=6,
        out_channels=len(OUTPUT_NAMES),   # 1 channel: ρ_GND
        modes=config['fno_modes'],
        width=config['fno_width'],
        n_layers=config['fno_layers']
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 FNO model created: {n_params:,} parameters")

    # Multi-GPU with DataParallel
    #if torch.cuda.device_count() > 1:
     #   print(f"➡ DataParallel on {torch.cuda.device_count()} GPUs")
      #  model = nn.DataParallel(model)
        # DataParallel splits the batch across the available GPUs
        # Gradients are computed on each GPU then aggregated on GPU 0
    #else:
     #   print("➡ 1 GPU")
    model = model.to(device)

    # ── Training ─────────────────────────────────────────────────────────
    print(f"🚀 Starting FNO training (lr={config['learning_rate']})...")
    trained_model, history = train_model(
        model, train_loader, val_loader, device, save_dir,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        output_ranges=output_ranges,
        modes=config['fno_modes'],
        width=config['fno_width'],
        n_layers=config['fno_layers'],
        history_suffix="_phase1"
    )

    # ── Test ─────────────────────────────────────────────────────────────
    print("🧪 Starting test...")
    test_model(
        trained_model, test_loader, device, save_dir,
        input_ranges, output_ranges, full_dataset, test_set
    )

    print("✅ Program completed successfully!")

    # ── Total time ───────────────────────────────────────────────────────
    end_time  = time.time()
    total_sec = end_time - start_time
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    print(f"\nTotal time: {h}h {m}m {s}s")