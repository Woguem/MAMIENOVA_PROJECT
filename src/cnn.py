# ==========================================================
# @author : Dr Yen Fred WOGUEM
# Project: Crystal Plasticity Deep Learning for Predicting Localized Deformation (MAMIE NOVA)
# Description: This script uses a fully convolutional network (CNN baseline) to predict the local
#              geometrically necessary dislocation density field (rho_GND) from CPFFT simulation
#              outputs. It loads both textured and non-textured structures.
# Based on: Olga Ibragimova et al., 2025
# Framework: PyTorch
# ==========================================================

import torch, gc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from torch.utils.tensorboard import SummaryWriter
import time
import csv
import torch.nn.functional as F
import sys

# ==========================================================
# Output channel: rho_GND only
# ==========================================================
OUTPUT_NAMES = [
    "rhognd"   # geometrically necessary dislocation density (m^-2)
]



# ==========================================================
# Reproducibility: environment configuration
# ==========================================================
os.environ['PYTHONHASHSEED'] = str(42)  # For Python hashing
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA (if GPU)

# Fix the random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# For CUDA (NVIDIA GPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 

start_time = time.time()


# ==========================================================
# 1. Define the Custom Dataset for Binary Files
# ==========================================================
class CrystalPlasticityDataset(Dataset):
    """
    Custom dataset for reading CPFFT binary output files.
    Each simulation folder contains:
    - 26 binary files (Outputs_0_j.bin) containing local properties
    - rho0.txt containing the initial dislocation density
    - MacroStrainStress.txt containing macroscopic properties
    """
    def __init__(self, base_dir, input_ranges, output_ranges, file_list=None, augment=False):
        super().__init__()
        self.base_dir = base_dir
        self.input_ranges = input_ranges
        self.output_ranges = output_ranges
        
        self.epsilon_range = input_ranges["epsilon"]
        self.rho0_range = input_ranges["rho0"]
        
        self.augment = augment
        
        self.file_list = file_list
        self.file_generator = self._load_all_file_paths()
        self._length = sum(1 for _ in self._load_all_file_paths()) if file_list is None else len(file_list)
        
        
    def normalize(self, x, r):
        denom = r[1] - r[0] 
        if denom == 0:
            raise ValueError(f"Invalid range {r}")
        return 2.0 * (x - r[0]) / denom - 1.0
    
    def rotate_180_x(self, x, y):
        """180° rotation de 180° arround x axis (haut ↔ bas)."""
        x_rot = np.flip(x, axis=1).copy()  # Inverse l'axe y (haut ↔ bas)
        y_rot = np.flip(y, axis=1).copy()
        return x_rot, y_rot

    def rotate_180_y(self, x, y):
        """180° rotation de 180° arround y axis (gauche ↔ droite)."""
        x_rot = np.flip(x, axis=2).copy()  # Inverse l'axe x (gauche ↔ droite)
        y_rot = np.flip(y, axis=2).copy()
        return x_rot, y_rot
    
    
    def visualize_sample(self, x, y, x_rot_x, y_rot_x, x_rot_y, y_rot_y, x_rot_yx, y_rot_yx):
        """Visualize original and transformed samples."""

        # Check the input dimensions
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("x and y should be 3D arrays with shape (channels, height, width)")

        fig, axes = plt.subplots(4, 2, figsize=(12, 36))  # 4 rows, one per transformation

        # Original
        axes[0, 0].imshow(x[2, :, :], cmap='viridis')
        axes[0, 0].set_title("Original (Input - Quaternions)")
        axes[0, 1].imshow(y[0, :, :], cmap='viridis')
        axes[0, 1].set_title("Original (Output - rho_GND)")

        # Rotated X
        axes[1, 0].imshow(x_rot_x[2, :, :], cmap='viridis')
        axes[1, 0].set_title("Rotated X (Input - Quaternions)")
        axes[1, 1].imshow(y_rot_x[0, :, :], cmap='viridis')
        axes[1, 1].set_title("Rotated X (Output - rho_GND)")

        # Rotated Y
        axes[2, 0].imshow(x_rot_y[2, :, :], cmap='viridis')
        axes[2, 0].set_title("Rotated Y (Input - Quaternions)")
        axes[2, 1].imshow(y_rot_y[0, :, :], cmap='viridis')
        axes[2, 1].set_title("Rotated Y (Output - rho_GND)")

        # Double Rotated X
        axes[3, 0].imshow(x_rot_yx[2, :, :], cmap='viridis')
        axes[3, 0].set_title("Double Rotated X (Input - Quaternions)")
        axes[3, 1].imshow(y_rot_yx[0, :, :], cmap='viridis')
        axes[3, 1].set_title("Double Rotated X (Output - rho_GND)")


        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "sample_transformations.png"))
        plt.close()


                
    def _load_all_file_paths(self):
        """Generate file paths for all simulations on the fly, excluding Outputs_0_000.bin for each structure."""
        sim_dirs = sorted([d for d in os.listdir(self.base_dir) if d.startswith("structure_")])
        for sim_dir_name in sim_dirs:
            sim_dir = os.path.join(self.base_dir, sim_dir_name)
            outputs_dir = os.path.join(sim_dir, "Outputs")

            if not os.path.exists(outputs_dir):
                raise FileNotFoundError(f"The Outputs directory does not exist for simulation: {sim_dir}")

            # List all .bin files in Outputs, excluding Outputs_0_000.bin
            bin_files = []
            for f in os.listdir(outputs_dir):
                if f.startswith("Outputs_0_") and f.endswith(".bin") and f != "Outputs_0_000.bin":
                    bin_files.append(os.path.join(outputs_dir, f))

            if not bin_files:
                raise FileNotFoundError(f"No valid binary files found in Outputs directory: {outputs_dir}")

            bin_files.sort()
            for bin_file in bin_files:
                yield bin_file
                
                
    # def _load_all_file_paths(self):
    #     """Generate file paths for all simulations on the fly, loading only Outputs_0_002.bin for each structure."""
    #     sim_dirs = sorted([d for d in os.listdir(self.base_dir) if d.startswith("structure_")])
    #     for sim_dir_name in sim_dirs:
    #         sim_dir = os.path.join(self.base_dir, sim_dir_name)
    #         outputs_dir = os.path.join(sim_dir, "Outputs")

    #         if not os.path.exists(outputs_dir):
    #             raise FileNotFoundError(f"The Outputs directory does not exist for simulation: {sim_dir}")

    #         target_file = "Outputs_0_002.bin"
    #         bin_file_path = os.path.join(outputs_dir, target_file)

    #         if os.path.exists(bin_file_path):
    #             yield bin_file_path



    def get_structure_name(self, bin_file):
        """Extract the structure name from the binary file path."""
        parts = bin_file.split(os.sep)
        for part in parts:
            if part.startswith("structure_"):
                return part
        raise ValueError(f"Could not extract structure name from {bin_file}")

    def filter_files_by_structures(self, structures):
        """Filter the dataset to only include files from the specified structures."""
        filtered_files = []
        for bin_file in self._load_all_file_paths():
            structure_name = self.get_structure_name(bin_file)
            if structure_name in structures:
                filtered_files.append(bin_file)
        return filtered_files

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return self._length
    

    def __getitem__(self, idx):
        """
        Load a sample by index.
        """
        if self.file_list is not None:
            bin_file = self.file_list[idx]
        else:
            gen = self._load_all_file_paths()
            for _ in range(idx):
                next(gen)
            bin_file = next(gen)
        x, y = self.load_data_pair(bin_file)
        
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Apply augmentation if activated
        if self.augment:
            # Return three verison (original + transformed)
            x_original = torch.from_numpy(x)
            y_original = torch.from_numpy(y)

            x_rot_x, y_rot_x = self.rotate_180_x(x, y)
            x_rot_y, y_rot_y = self.rotate_180_y(x, y)
            x_rot_yx, y_rot_yx = self.rotate_180_x(x_rot_y, y_rot_y)

            x_rot_x = torch.from_numpy(x_rot_x)
            y_rot_x = torch.from_numpy(y_rot_x)
            
            x_rot_y = torch.from_numpy(x_rot_y)
            y_rot_y = torch.from_numpy(y_rot_y)
            
            x_rot_yx = torch.from_numpy(x_rot_yx)
            y_rot_yx = torch.from_numpy(y_rot_yx)
            
            
            # Visualize only the first sample
            if idx == 5:
                self.visualize_sample(x_original, y_original, x_rot_x, y_rot_x, x_rot_y, y_rot_y, x_rot_yx, y_rot_yx)


            return [
                (x_original, y_original, bin_file),
                (x_rot_x, y_rot_x, bin_file),
                (x_rot_y, y_rot_y, bin_file),
                (x_rot_yx, y_rot_yx, bin_file)]
        else:
            # Return only original version 
            return torch.from_numpy(x), torch.from_numpy(y), bin_file


    def read_fortran_record(self, f, shape, dtype=np.float32):
        """Read a Fortran unformatted record from a binary file"""
        f.read(4)  # Skip leading record marker
        data = np.fromfile(f, dtype=dtype, count=np.prod(shape))
        f.read(4)  # Skip trailing record marker
        return data.reshape(shape, order='F')
    
    def euler_to_quat(self, phi1, Phi, phi2):
        """
        Convert Euler angles (φ₁, Φ, φ₂) to quaternions.
        Parameters:
        - phi1: np.array of shape (H, W) containing φ₁ in radians
        - Phi: np.array of shape (H, W) containing Φ in radians
        - phi2: np.array of shape (H, W) containing φ₂ in radians
        Returns:
        - quaternions: np.array of shape (4, H, W) containing quaternions
        """
        # Calculate intermediate angles
        Ang_plus = (phi1 + phi2) / 2
        Ang_moins = (phi2 - phi1) / 2
        sbeta = Phi / 2
        # Calculate quaternion components
        q1 = np.cos(sbeta) * np.cos(Ang_plus)
        q2 = np.sin(sbeta) * np.cos(Ang_moins)
        q3 = -np.sin(sbeta) * np.sin(Ang_moins)
        q4 = np.cos(sbeta) * np.sin(Ang_plus)
        # Normalize quaternions
        norm = np.sqrt(q1**2 + q2**2 + q3**2 + q4**2)
        # Stack quaternions
        quaternions = np.stack([q1, q2, q3, q4], axis=0) / norm
        # Ensure q4 is positive for consistency
        quaternions[:, q4 < 0] *= -1
        return quaternions
    

    def load_data_pair(self, bin_file):
        """
        Load input and output data from a binary file.
        Inputs: epsilon_macro_11, rho0, quaternion from Euler angles
        Output: local GND dislocation density field rho_GND
        Parameters:
        - bin_file: path to the binary file
        Returns:
        - X: input features (6, H, W)  # 6 channels: 1 (epsilon), 1 (rho0), 4 (quaternion)
        - Y: output labels (1, H, W)   # 1 channel: rho_GND
        """
        # Extract simulation index and step from file name
        file_name = os.path.basename(bin_file)
        eps_macro_str = file_name.split('_')[2].split('.')[0]
        epsilon_macro_11_from_file = float("0." + eps_macro_str)

        # Read rho0 from rho0.txt file
        rho0_file = os.path.join(os.path.dirname(bin_file), 'rho0.txt')
        if not os.path.isfile(rho0_file):
            raise FileNotFoundError(f"❌ Missing rho0.txt in {os.path.dirname(bin_file)}")
        with open(rho0_file, 'r') as f:
            text = f.read().replace('d', 'e').replace('D', 'e').strip()
        if not text:
            raise ValueError(f"❌ rho0.txt is empty in {os.path.dirname(bin_file)}")
        try:
            rho0 = float(text)
        except ValueError as e:
            raise ValueError(f"❌ Invalid content in {rho0_file}: '{text}'") from e

        # Read binary file
        nx, ny, nz = 128, 128, 1
        # All records are read in file order to reach the needed fields;
        # only rho_GND and the Euler angles are used below.
        with open(bin_file, 'rb') as f:
            # Read strain and stress components
            eps11 = self.read_fortran_record(f, (nx, ny, nz))
            eps22 = self.read_fortran_record(f, (nx, ny, nz))
            eps33 = self.read_fortran_record(f, (nx, ny, nz))
            eps32 = self.read_fortran_record(f, (nx, ny, nz))
            eps31 = self.read_fortran_record(f, (nx, ny, nz))
            eps12 = self.read_fortran_record(f, (nx, ny, nz))
            sig11 = self.read_fortran_record(f, (nx, ny, nz))
            sig22 = self.read_fortran_record(f, (nx, ny, nz))
            sig33 = self.read_fortran_record(f, (nx, ny, nz))
            sig32 = self.read_fortran_record(f, (nx, ny, nz))
            sig31 = self.read_fortran_record(f, (nx, ny, nz))
            sig12 = self.read_fortran_record(f, (nx, ny, nz))
            rhognd = self.read_fortran_record(f, (nx, ny, nz))
            rhossd = self.read_fortran_record(f, (nx, ny, nz))
            phi1 = self.read_fortran_record(f, (nx, ny, nz))
            phi = self.read_fortran_record(f, (nx, ny, nz))  
            phi2 = self.read_fortran_record(f, (nx, ny, nz))

        # Compute macroscopic stress and strain from local fields
        epsilon_macro_11 = epsilon_macro_11_from_file   # mean of local eps11
        #sigma_macro_11   = np.mean(sig11)   # mean of local sig11

        # Extract a 2D slice (first layer in z)
        phi1_2d = phi1[:, :, 0]
        phi_2d = phi[:, :, 0]
        phi2_2d = phi2[:, :, 0]

        # Convert Euler angles to quaternions
        quaternions = self.euler_to_quat(phi1_2d, phi_2d, phi2_2d)

        # Prepare input features (X)
        H, W = ny, nx

        # Normalize epsilon_macro_11 to [-1, 1]
        epsilon_norm = 2 * (epsilon_macro_11 - self.epsilon_range[0]) / (self.epsilon_range[1] - self.epsilon_range[0]) - 1
        epsilon_M = np.full((1, H, W), epsilon_norm, dtype=np.float32)

        # Normalize rho0 (log scale) to [-1, 1]
        rho0_log = np.log10(rho0 + 1.0)
        rho0_log_min = np.log10(self.rho0_range[0] + 1.0)
        rho0_log_max = np.log10(self.rho0_range[1] + 1.0)
        rho0_norm = 2 * (rho0_log - rho0_log_min) / (rho0_log_max - rho0_log_min) - 1
        rho0_M = np.full((1, H, W), rho0_norm, dtype=np.float32)

        # Concatenate epsilon, rho0, and quaternions
        X = np.concatenate([
            epsilon_M,  # (1, H, W)
            rho0_M,      # (1, H, W)
            quaternions  # (4, H, W)
        ], axis=0)  # Final shape: (6, H, W)
        
        # rho_GND: log10 transform before normalisation (spans several decades)
        rhognd = np.log10(rhognd + 1.0)
        rhognd_norm = self.normalize(rhognd, self.output_ranges["rhognd"])

        # Single output channel (first z layer -> 2D)
        Y = rhognd_norm[np.newaxis, :, :, 0]  # Shape: (1, H, W)
        
        
        return X.astype(np.float32), Y.astype(np.float32)

    
def collate_fn_train(batch):
    all_samples = []
    for item in batch:
        all_samples.extend(item)

    x_batch = torch.stack([sample[0] for sample in all_samples])
    y_batch = torch.stack([sample[1] for sample in all_samples])
    files   = [sample[2] for sample in all_samples]

    return x_batch, y_batch, files


def collate_fn_val(batch):
    x_batch = torch.stack([sample[0] for sample in batch])
    y_batch = torch.stack([sample[1] for sample in batch])
    files   = [sample[2] for sample in batch]
    return x_batch, y_batch, files




# ==========================================================
# 2. Define the Mamie Nova Model
# ==========================================================

class SimpleUNetBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True):
        super(SimpleUNetBlockEncoder, self).__init__()

        if use_batchnorm:
            Norm = lambda c: nn.BatchNorm2d(c)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode="circular", stride=stride)
        self.norm1 = Norm(out_channels)
        #self.act = nn.ReLU()
        self.act = nn.Tanh() # nn.Tanh(), nn.Sigmoid()

        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, padding_mode="circular", stride=1)
        # self.norm2 = Norm(out_channels)

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        # out = self.act(self.norm2(self.conv2(out)))
        return out

# class SimpleUNetBlockDecoder(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True):
#         super(SimpleUNetBlockDecoder, self).__init__()

#         if use_batchnorm:
#             Norm = lambda c: nn.BatchNorm2d(c)
       

#         self.tconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=7, padding=2, padding_mode="circular", stride=stride)
#         self.norm1 = Norm(out_channels)
#         self.act = nn.ReLU()

#         # self.tconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding="same", stride=1)
#         # self.norm2 = Norm(out_channels)

#     def forward(self, x):
#         out = self.act(self.norm1(self.tconv1(x)))
#         # out = self.act(self.norm2(self.tconv2(out)))
#         return out


class SimpleUNetModel(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, use_batchnorm=True):
        super(SimpleUNetModel, self).__init__()

        #self.act = nn.ReLU()
        self.act = nn.Tanh() # nn.Tanh(), nn.Sigmoid()

        # Encoder
        self.encoder_first = nn.Conv2d(in_channels, 16, kernel_size=7, padding=3, padding_mode="circular", stride=1)
        
        self.encoder_block1 = SimpleUNetBlockEncoder(16, 32, stride=1, use_batchnorm=use_batchnorm)
        self.encoder_block2 = SimpleUNetBlockEncoder(32, 64, stride=1, use_batchnorm=use_batchnorm)
        self.encoder_block3 = SimpleUNetBlockEncoder(64, 128, stride=1, use_batchnorm=use_batchnorm)
        self.encoder_block4 = SimpleUNetBlockEncoder(128, 256, stride=1, use_batchnorm=use_batchnorm)
        self.encoder_block5 = SimpleUNetBlockEncoder(256, 512, stride=1, use_batchnorm=use_batchnorm)
        self.encoder_block6 = SimpleUNetBlockEncoder(512, out_channels, stride=1, use_batchnorm=use_batchnorm)
    
        

        # # Decoder
        # self.decoder_block1 = SimpleUNetBlockDecoder(1024, 512, stride=1, use_batchnorm=use_batchnorm)
        # self.decoder_block2 = SimpleUNetBlockDecoder(512, 256, stride=1, use_batchnorm=use_batchnorm)
        # self.decoder_block3 = SimpleUNetBlockDecoder(256, 128, stride=1, use_batchnorm=use_batchnorm)
        # self.decoder_block4 = SimpleUNetBlockDecoder(128, 64, stride=1, use_batchnorm=use_batchnorm)
        # self.decoder_block5 = SimpleUNetBlockDecoder(64, 32, stride=1, use_batchnorm=use_batchnorm)
        # self.decoder_block6 = SimpleUNetBlockDecoder(32, 16, stride=1, use_batchnorm=use_batchnorm)

        # # Out
        # self.out = nn.ConvTranspose2d(16, out_channels, kernel_size=7, padding=2, padding_mode="circular", stride=1)

        # Activation finale
        self.final_act = nn.Tanh()

    def forward(self, x):
        # Encoder
        inp = self.encoder_first(x)
        
        e1 = self.encoder_block1(inp)
        e2 = self.encoder_block2(e1)
        e3 = self.encoder_block3(e2)
        e4 = self.encoder_block4(e3)
        e5 = self.encoder_block5(e4)
        e6 = self.encoder_block6(e5)
       
        
        
        # # Decoder
        # d1 = self.decoder_block1(e6)
        # d2 = self.decoder_block2(d1)
        # d3 = self.decoder_block3(d2)
        # d4 = self.decoder_block4(d3)
        # d5 = self.decoder_block5(d4)
        # d6 = self.decoder_block6(d5)

        # Out

        out = self.final_act(e6)

        return out

def MamieNovaUNETModel(in_channels=6, out_channels=1, use_batchnorm=True):
    model = SimpleUNetModel(in_channels, out_channels, use_batchnorm=use_batchnorm)
    return model





   
# ==========================================================
# 3. Training Function
# ==========================================================






def train_model(model, train_loader, val_loader, device, save_dir, epochs, lr, output_ranges, history_suffix=''):
    """
    Train the CNN model for predicting the rho_GND field.
    Parameters:
    - model: CNN model
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - device: device to use for training ('cuda' or 'cpu')
    - epochs: number of training epochs
    - lr: learning rate
    Returns:
    - model: trained model
    - history: training history
    """
    
    # criterion_sig = WeightedSmoothL1([1.0, 1.0, 1.0])
    #criterion_eps = WeightedSmoothL1([1.0, 1.0, 1.0, 1.0])
    # criterion_rho = WeightedSmoothL1([1.0, 1.0])
    
    # lambda_grad = 2
    # lambda_rel = 2

    criterion = nn.MSELoss(reduction="mean")
    
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4)


    
    # scheduler = OneCycleLR(
    # optimizer,
    # max_lr=1e-4,  
    # epochs=epochs,
    # steps_per_epoch=len(train_loader),
    # pct_start=0.2,  
    # div_factor=25,   
    # final_div_factor=1e4,) 
    
    history = {"epoch": [], "train_loss": [], "val_loss": []}#, "learning_rate": []}
    

    

    
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")
    history_csv_path = os.path.join(save_dir, f"training_history{history_suffix}.csv")

    
    # Initialize the SummaryWriter for TensorBoard
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_logs'))

    # Check if the directory exists, otherwise create it : save loss and accuracy
    os.makedirs(save_dir, exist_ok=True)
    


    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        #y_true_train, y_pred_train = [], []

        for x_batch, y_batch, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)

            # MSE on the normalised rho_GND channel
            loss = criterion(y_pred, y_batch)


            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= (4*len(train_loader.dataset)) 

        
        
        
        
        # For tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        
        
        # ---- Validation every 3 epochs ----
        val_loss = None
    
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for x_val, y_val, _ in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_pred = model(x_val)

                    loss = criterion(y_pred, y_val)
                    
                    val_loss += loss.item() * x_val.size(0)
                    
            val_loss /= len(val_loader.dataset)
            
            
            # Add scheduler
            # if scheduler is not None:
            #    scheduler.step(val_loss)

            # For tensorboard
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            
            
            # Save best model based on validation MSE
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

                torch.save(
                    {
                        'epoch': epoch + 1,
                        'state_dict': state_dict_to_save, 
                        'val_loss': best_val_loss
                    },best_model_path)
                                
                print(f"✅ New best model weights saved at epoch {epoch+1} with val loss={val_loss:.6f}")
                
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
                
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()        

        # Save in history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        


        # Save history to CSV every 3 epochs (overwrite)
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            history_df = pd.DataFrame(history)
            history_df.to_csv(history_csv_path, index=False)
            print(f"✅ Training history saved to {history_csv_path} (overwritten)")

        # Display progress
        if val_loss is not None:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")

    # Load the best model weights at the end of training
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        state_dict = checkpoint["state_dict"]

        # Load into the right object (model.module if DataParallel)
        target = model.module if isinstance(model, nn.DataParallel) else model

        # In case an old checkpoint still has the "module." prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        target.load_state_dict(state_dict, strict=True)
        model.to(device)

        print(f"✅ Best model weights loaded from epoch {checkpoint['epoch']} with val Loss={checkpoint['val_loss']:.4f}")
    

    # Final plot and save (same file, overwritten)
    training_curves_path = os.path.join(save_dir, "training_curves.png")
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=history_df, x="epoch", y="train_loss", label="Train Loss")
    sns.lineplot(data=history_df, x="epoch", y="val_loss", label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(training_curves_path)
    plt.close()
    print(f"✅ Training curves saved to {training_curves_path}")
    
    

    
    # Close SummaryWriter
    writer.close()

    return model, history_df


# ==========================================================
# 4. Test Function
# ==========================================================

def count_outliers(data):
    """Count the number of outliers in the data."""
    if len(data) == 0:
        return 0, 0, 0, 0
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    num_outliers = len(outliers)
    total_points = len(data)
    percentage_outliers = (num_outliers / total_points) * 100 if total_points > 0 else 0
    return num_outliers, percentage_outliers, lower_bound, upper_bound


def test_model(model, test_loader, device, save_dir, input_ranges, output_ranges, full_dataset, test_set):
    """
    Test the model on unseen data, calculate RMSE and MAE errors for each output channel,
    and visualize results with box plots.

    Parameters:
    - model: trained CNN model
    - test_loader: DataLoader for test data
    - device: device to use for testing ('cuda' or 'cpu')
    - save_dir: directory to save test results
    - output_ranges: dictionary containing normalization ranges for each output channel
    - full_dataset: full dataset object
    - test_set: test dataset object
    """

    model.to(device)
    model.eval()
    output_names = OUTPUT_NAMES

    rmse_errors = {name: [] for name in output_names}
    mae_errors  = {name: [] for name in output_names}
    structure_data = {}

    sample_idx = 0  # ✅ global index over all samples

    with torch.no_grad():
        for x_test, y_test, _ in test_loader:   # ✅ no idx here
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_pred = model(x_test)
            y_true = y_test.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            # Denormalisation
            y_true_denorm = np.zeros_like(y_true)
            y_pred_denorm = np.zeros_like(y_pred)
            for i, name in enumerate(output_names):
                r = output_ranges[name]
                if name in ["rhossd", "rhognd"]:
                    # Denormalise, then apply the exponential to return to the linear scale
                    y_true_denorm[:, i] = (y_true[:, i] * ((r[1] - r[0]) / 2) + (r[0] + r[1]) / 2)
                    y_true_denorm[:, i] = 10 ** y_true_denorm[:, i] - 1

                    y_pred_denorm[:, i] = (y_pred[:, i] * ((r[1] - r[0]) / 2) + (r[0] + r[1]) / 2)
                    y_pred_denorm[:, i] = 10 ** y_pred_denorm[:, i] - 1
                else:
                    y_true_denorm[:, i] = y_true[:, i] * ((r[1] - r[0]) / 2) + (r[0] + r[1]) / 2
                    y_pred_denorm[:, i] = y_pred[:, i] * ((r[1] - r[0]) / 2) + (r[0] + r[1]) / 2

            # RMSE / MAE
            for i, name in enumerate(output_names):
                for j in range(y_true.shape[0]):
                    true_map = y_true_denorm[j, i]
                    pred_map = y_pred_denorm[j, i]
                    rmse_errors[name].append(np.sqrt(np.mean((true_map - pred_map)**2)))
                    mae_errors[name].append(np.mean(np.abs(true_map - pred_map)))

            # ✅ Per-structure storage with the global index
            for j in range(y_true.shape[0]):
                bin_file       = test_set.file_list[sample_idx]   # ← correct
                structure_name = full_dataset.get_structure_name(bin_file)
                sample_idx    += 1

                if structure_name not in structure_data:
                    structure_data[structure_name] = {
                        "Strain":    [],
                        "Quaternion": [],
                        "True":      {name: [] for name in output_names},
                        "Predicted": {name: [] for name in output_names}
                    }

                strain_val = (x_test[j, 0, 0, 0].cpu().numpy()
                              * ((input_ranges["epsilon"][1] - input_ranges["epsilon"][0]) / 2)
                              + (input_ranges["epsilon"][0]  + input_ranges["epsilon"][1])  / 2)

                structure_data[structure_name]["Strain"].append(strain_val)
                structure_data[structure_name]["Quaternion"].append(x_test[j, 2:6].cpu().numpy())

                for i, name in enumerate(output_names):
                    structure_data[structure_name]["True"][name].append(y_true_denorm[j, i])
                    structure_data[structure_name]["Predicted"][name].append(y_pred_denorm[j, i])

    # ✅ Clean saving without nested dicts
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
        print(f"✅ Saved {save_path}")


    # Save outliers information
    output_file = os.path.join(save_dir, "test_results_info.txt")
    with open(output_file, 'w') as f:
        f.write("=== SUMMARY STATISTICS ===\n")
        for name in output_names:
            rmse_data = rmse_errors[name]
            mae_data = mae_errors[name]

            if len(rmse_data) > 0 and len(mae_data) > 0:
                rmse_mean = np.mean(rmse_data)
                rmse_q1, rmse_q3 = np.percentile(rmse_data, [25, 75])
                mae_mean = np.mean(mae_data)
                mae_q1, mae_q3 = np.percentile(mae_data, [25, 75])
                f.write(f"--- {name} ---\n")
                f.write(f"RMSE: Mean = {rmse_mean:.4f}, Q1 = {rmse_q1:.4f}, Q3 = {rmse_q3:.4f}\n")
                f.write(f"MAE: Mean = {mae_mean:.4f}, Q1 = {mae_q1:.4f}, Q3 = {mae_q3:.4f}\n\n")
            else:
                f.write(f"--- {name} ---\n")
                f.write("No data available for RMSE and MAE calculations.\n\n")

        f.write("\n=== RMSE OUTLIERS INFORMATION ===\n")
        for name in output_names:
            rmse_vals = rmse_errors[name]
            if len(rmse_vals) > 0:
                num_outliers, percentage_outliers, lower_bound, upper_bound = count_outliers(rmse_vals)
                f.write(f"RMSE {name}: Total points = {len(rmse_vals)}, Outliers = {num_outliers} "
                        f"({percentage_outliers:.2f}%), Lower bound = {lower_bound:.4f}, "
                        f"Upper bound = {upper_bound:.4f}\n")
            else:
                f.write(f"RMSE {name}: No data available for outlier calculations.\n")

        f.write("\n=== MAE OUTLIERS INFORMATION ===\n")
        for name in output_names:
            mae_vals = mae_errors[name]
            if len(mae_vals) > 0:
                num_outliers, percentage_outliers, lower_bound, upper_bound = count_outliers(mae_vals)
                f.write(f"MAE {name}: Total points = {len(mae_vals)}, Outliers = {num_outliers} "
                        f"({percentage_outliers:.2f}%), Lower bound = {lower_bound:.4f}, "
                        f"Upper bound = {upper_bound:.4f}\n")
            else:
                f.write(f"MAE {name}: No data available for outlier calculations.\n")

    
    rmse_data = []
    mae_data = []
    for name in output_names:
        rmse_data.append(rmse_errors[name])
        mae_data.append(mae_errors[name])

    # Plot box plots for RMSE errors
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=rmse_data)
    plt.xticks(ticks=range(len(output_names)), labels=output_names, rotation=45)
    plt.ylabel('RMSE')
    plt.title('Distribution of RMSE Errors for Each Output Channel')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rmse_boxplots_all_channels.png"))
    plt.close()

    # Plot box plots for MAE errors
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=mae_data)
    plt.xticks(ticks=range(len(output_names)), labels=output_names, rotation=45)
    plt.ylabel('MAE')
    plt.title('Distribution of MAE Errors for Each Output Channel')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mae_boxplots_all_channels.png"))
    plt.close()
    


    print(f"✅ RMSE and MAE box plots saved in {save_dir}")
    print(f"✅ Outliers information saved in {output_file}")




def calculate_output_ranges(base_dir):
    output_ranges = {name: {"min": float('inf'), "max": float('-inf')} for name in OUTPUT_NAMES}

    sim_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("structure_")])

    for sim in sim_dirs:
        outputs_dir = os.path.join(base_dir, sim, "Outputs")
        bin_files = sorted(f for f in os.listdir(outputs_dir) if f.startswith("Outputs_0_") and f.endswith(".bin"))

        for bf in bin_files:
            with open(os.path.join(outputs_dir, bf), "rb") as f:
                nx, ny, nz = 128, 128, 1

                eps11 = read_fortran_record(f, (nx, ny, nz))
                eps22 = read_fortran_record(f, (nx, ny, nz))
                eps33 = read_fortran_record(f, (nx, ny, nz))
                _ = read_fortran_record(f, (nx, ny, nz))  # eps32
                _ = read_fortran_record(f, (nx, ny, nz))  # eps31
                eps12 = read_fortran_record(f, (nx, ny, nz))

                sig11 = read_fortran_record(f, (nx, ny, nz))
                sig22 = read_fortran_record(f, (nx, ny, nz))
                _ = read_fortran_record(f, (nx, ny, nz))  # sig33
                _ = read_fortran_record(f, (nx, ny, nz))  # sig32
                _ = read_fortran_record(f, (nx, ny, nz))  # sig31
                sig12 = read_fortran_record(f, (nx, ny, nz))

                rhognd = read_fortran_record(f, (nx, ny, nz))
                rhossd = read_fortran_record(f, (nx, ny, nz))

                rhognd = np.log10(rhognd + 1.0)

                # Update rho_GND min/max
                output_ranges["rhognd"]["min"] = min(output_ranges["rhognd"]["min"], np.min(rhognd))
                output_ranges["rhognd"]["max"] = max(output_ranges["rhognd"]["max"], np.max(rhognd))

    # Conversion en tuples (min, max)
    output_ranges = {name: (vals["min"], vals["max"]) for name, vals in output_ranges.items()}

    return output_ranges




def read_fortran_record(f, shape, dtype=np.float32):
    """Read a Fortran unformatted record from a binary file"""
    f.read(4)  # Skip leading record marker
    data = np.fromfile(f, dtype=dtype, count=np.prod(shape))
    f.read(4)  # Skip trailing record marker
    return data.reshape(shape, order='F')





# ==========================================================
# 5. Main Application
# ==========================================================

if __name__ == "__main__":
    """
    Main script to train and test the CNN baseline on CPFFT binary data (rho_GND only).
    - Splits dataset into train/val/test by structure (random split)
    - Trains model with MSE loss on the normalised log10(rho_GND + 1) field
    - Evaluates with RMSE and MAE (physical units, m^-2)
    - Saves per-structure predictions (.npz) and error box plots
    """

    # ============================================================
    # 0. File lists
    # ============================================================

    base_dir = os.getenv('DATASET_PATH')
    save_dir = os.getenv('RESULTS_PATH')
    # Verify paths
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"❌ The simulation base directory does not exist: {base_dir}")
    os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist
    if not os.access(save_dir, os.W_OK):
        raise PermissionError(f"❌ Cannot write to results directory: {save_dir}")
    print(f"📁 Dataset path: {base_dir}")
    print(f"💾 Results will be saved to: {save_dir}")

    # ============================================================
    # 1. Ranges for normalization [-1, 1] of scalar features
    # ============================================================
 
    input_ranges = {
    "epsilon": (0.0, 0.05), # macroscopic strain
    "rho0": (1e10, 1e14)} # initial dislocation density
    output_ranges = calculate_output_ranges(base_dir)  # Dynamically calculate the range of log10(rho_GND + 1)
    print(f"Output range: {output_ranges}")

    # ============================================================
    # 2. Configuration
    # ============================================================
    config = {
        'batch_size': 12,
        'epochs': 20, 
        'learning_rate': 1e-4,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1
    }

    # ============================================================
    # 3. Create dataset with both textured and non-textured structures
    # ============================================================
    full_dataset = CrystalPlasticityDataset(base_dir, input_ranges=input_ranges,output_ranges=output_ranges)
    
    if len(full_dataset) == 0:
        raise ValueError("❌ No data loaded. Check your file paths and data structure.")
    print(f"📁 Loading {len(full_dataset)} file pairs from both textured and non-textured structures")

    # ============================================================
    # 4. Split into train/val/test by structure (80/10/10) - random split of the structures
    # ============================================================
    # List all structures
    sim_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("structure_")])
    random.shuffle(sim_dirs)  # Random shuffle of the structures

    n_total_structures = len(sim_dirs)
    n_train_structures = int(config['train_ratio'] * n_total_structures)
    n_val_structures = int(config['val_ratio'] * n_total_structures)
    n_test_structures = n_total_structures - n_train_structures - n_val_structures

    # Select Random structures
    train_structures = sim_dirs[:n_train_structures]
    val_structures = sim_dirs[n_train_structures:n_train_structures + n_val_structures]
    test_structures = sim_dirs[n_train_structures + n_val_structures:]

    # print(f"📁 Train structures: {train_structures}")
    # print(f"📁 Validation structures: {val_structures}")
    # print(f"📁 Test structures: {test_structures}")

    # Filter .bin file for selected structures
    train_files = full_dataset.filter_files_by_structures(train_structures)
    val_files = full_dataset.filter_files_by_structures(val_structures)
    test_files = full_dataset.filter_files_by_structures(test_structures)

    # Create separate datasets
    train_set = CrystalPlasticityDataset(
                                            base_dir,
                                            input_ranges,
                                            output_ranges,
                                            file_list=train_files,
                                             augment=True)
                                            
    
    val_set = CrystalPlasticityDataset(
                                                    base_dir,
                                                    input_ranges,
                                                    output_ranges,
                                                     file_list=val_files,
                                                      augment=False)
                                                     
                                      
    test_set = CrystalPlasticityDataset(
                                        base_dir,
                                        input_ranges,
                                        output_ranges,
                                        file_list=test_files,
                                         augment=False)
                                        

    # ============================================================
    # 5. DataLoaders
    # ============================================================
    default_num_workers = 12

     
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn_train, num_workers=default_num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn_val, num_workers=default_num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,  collate_fn=collate_fn_val, num_workers=default_num_workers, pin_memory=True) # MANDATORY: sample_idx in test_model depends on the order

    # ============================================================
    # 6. Device
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device used: {device}")

    # ============================================================
    # 7. Model
    # ============================================================
    model = MamieNovaUNETModel(in_channels=6, out_channels=len(OUTPUT_NAMES), use_batchnorm=True)  # 1 channel: rho_GND
    print(f"🧠 Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # --- Activation multi-GPU automatique ---
    if torch.cuda.device_count() > 1:
        print(f"➡ Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    else:
        print("➡ Using a single GPU")
    model = model.to(device)  # Displace model on GPU device

    # ============================================================
    # 8. Training
    # ============================================================
    print("🚀 Start of training (phase 1, lr=1e-4)...")
    trained_model, history = train_model(
        model, train_loader, val_loader, device, save_dir,
        epochs=config['epochs'], lr=config['learning_rate'], output_ranges=output_ranges, history_suffix="_phase1")
    
    # print("🔧 Start of fine-tuning (phase 2, lr=1e-4)...")
    # fine_tuned_model, history_ft = train_model(
    # trained_model, train_loader, val_loader, device, save_dir,
    # epochs=100, lr=1e-5, history_suffix="_phase2")

    # ============================================================
    # 9. Testing
    # ============================================================
    print("🧪 Start of testing...")
    test_metrics = test_model(
                        trained_model,
                        test_loader,
                        device,
                        save_dir,
                        input_ranges,
                        output_ranges,
                        full_dataset,
                        test_set
                    )

    print("✅ Program successfully completed!")

    end_time = time.time()
    total_sec = end_time - start_time

    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    seconds = int(total_sec % 60)

    print(f"\nTotal execution times : {hours}h {minutes}m {seconds}s")