# Hybrid FNO–CNN for GND-density prediction in polycrystalline aluminium

**Author:** Dr Yen Fred Woguem
**Project:** MAMIE NOVA — crystal-plasticity deep learning

Source code supporting the manuscript:

> *A hybrid Fourier neural operator–convolutional surrogate for predicting geometrically necessary dislocation density in polycrystalline aluminum.*

## Overview

The repository contains three independent training/evaluation scripts. All three predict the same quantity — the local **geometrically necessary dislocation (GND) density field, ρ_GND** — from the same inputs, so their results can be compared directly.

| Script | Model | Role in the manuscript |
|---|---|---|
| `fno_cnn.py` | Hybrid FNO–CNN: each block sums a spectral convolution, a 1×1 bypass and a local branch of four 3×3 circular convolutions | Proposed model |
| `fno.py` | Fourier neural operator: spectral convolution + 1×1 bypass, with residual connection | Baseline |
| `cnn.py` | Fully convolutional network (7×7 circular convolutions, no down-sampling) | Baseline |

## Inputs and output

Each sample is one strain step of one CPFFT simulation, on a 128 × 128 grid.

**Input — 6 channels**

| Channel | Quantity | Pre-processing |
|---|---|---|
| 0 | Applied macroscopic strain ε₁₁ (read from the file name, e.g. `Outputs_0_025.bin` → 0.025) | Normalised to [−1, 1] over [0, 0.05] |
| 1 | Initial dislocation density ρ₀ (from `rho0.txt`) | log₁₀(ρ₀ + 1), normalised to [−1, 1] over [10¹⁰, 10¹⁴] m⁻² |
| 2–5 | Local crystal orientation as a unit quaternion (q1–q4), computed from the Euler angles (φ₁, Φ, φ₂) | Hemisphere convention q4 ≥ 0 |

**Output — 1 channel**

| Channel | Quantity | Pre-processing |
|---|---|---|
| 0 | ρ_GND (m⁻²) | log₁₀(ρ_GND + 1), normalised to [−1, 1] using the global min/max computed over the whole dataset |

Test metrics (RMSE, MAE) are reported after inverting the normalisation and the log transform, i.e. in m⁻².

## Data format

Set `DATASET_PATH` to a directory containing simulation folders named `structure_*`:

```
dataset/
├── structure_0_nt/
│   └── Outputs/
│       ├── rho0.txt
│       ├── Outputs_0_000.bin   # ε = 0, used only for the normalisation range
│       ├── Outputs_0_002.bin
│       ├── Outputs_0_004.bin
│       └── ...
├── structure_1_nt/
│   └── Outputs/
│       └── ...
└── ...
```

Each `Outputs_0_YYY.bin` file is a Fortran-unformatted binary file (float32, 4-byte record markers) with 17 records of size 128 × 128 × 1, in this order:

| Records | Fields | Used |
|---|---|---|
| 1–6 | ε₁₁, ε₂₂, ε₃₃, ε₃₂, ε₃₁, ε₁₂ | No |
| 7–12 | σ₁₁, σ₂₂, σ₃₃, σ₃₂, σ₃₁, σ₁₂ | No |
| 13 | ρ_GND | Target |
| 14 | ρ_SSD | No |
| 15–17 | φ₁, Φ, φ₂ (radians) | Input (→ quaternions) |

`rho0.txt` contains a single number; Fortran exponents (`1.0d12`) are accepted.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch builds depend on your OS and CUDA version. If you need GPU support, first install the matching build from <https://pytorch.org/get-started/locally/>, then install the remaining packages.

## Usage

```bash
export DATASET_PATH="/absolute/path/to/dataset"
export RESULTS_PATH="/absolute/path/to/results"

python src/fno_cnn.py   # proposed hybrid model
python src/fno.py       # FNO baseline
python src/cnn.py       # CNN baseline
```

Use a separate `RESULTS_PATH` for each model, since all scripts write files with the same names.

Each script:

1. computes the ρ_GND normalisation range over the dataset;
2. splits the **structures** (not individual strain steps) at random into 80 % train, 10 % validation and 10 % test, with seed 42;
3. trains with MSE loss on the normalised target, using AdamW (lr = 1e-4, weight decay = 1e-4) and 4× data augmentation (the original map, two mirror images and a 180° in-plane rotation), validating every 3 epochs with early stopping;
4. evaluates the best checkpoint on the test structures.

## Main hyperparameters

All values are set in the `config` dictionary of each script.

## Outputs

These files are written to `RESULTS_PATH`:

| File | Content |
|---|---|
| `data_split.json` | Train / validation / test structure lists |
| `best_model.pth` | Best checkpoint (by validation loss) |
| `training_history_phase1.csv`, `training_curves.png` | Loss history |
| `tensorboard_logs/` | TensorBoard logs |
| `structure_*.npz` | Per test structure: `strain`, `quaternion`, `true_rhognd`, `pred_rhognd` (m⁻²) |
| `test_results_info.txt` | RMSE / MAE statistics and IQR outliers |
| `rmse_boxplots_all_channels.png`, `mae_boxplots_all_channels.png` | Error distributions |

## Reproducibility

The scripts fix the Python, NumPy and PyTorch seeds (42) and enable deterministic cuDNN. Deterministic algorithms are enforced in `fno_cnn.py` and `cnn.py`. Bitwise reproducibility is still not guaranteed across different hardware, CUDA/cuDNN or library versions.

For each run, record:

- the commit of this repository;
- the Python, PyTorch, NumPy and CUDA versions;
- the dataset version;
- `data_split.json` and `best_model.pth`.

## Data availability

- **Dataset:** TODO — add the DOI or URL once the dataset is deposited, or describe the access conditions.

## Citation

If you use this code, please cite the accompanying manuscript : ......

## License

MIT — see [`LICENSE`](LICENSE).
