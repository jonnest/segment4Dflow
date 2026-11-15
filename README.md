# Segment4DFlow

Deep-learning segmentation of blood vessels in **4D Flow MRI** data.

The repository trains and applies U-Net-type networks (2D and 3D) that turn
4D Flow MRI image volumes into binary vessel masks. The current focus is
**brain vessels** (`SegmentBrainVessels`). An earlier 2D aorta model
(`SegmentAorta`) is also supported. The network takes two input channels:

| Channel      | Meaning                                                    |
|--------------|------------------------------------------------------------|
| `magnitude`  | MRI magnitude image (anatomy)                              |
| `cd`         | Complex-difference / flow-derived image (highlights flow)  |
| `velocity`   | Velocity image (used by the older 2D two-channel models)   |

After inference, the predicted mask is post-processed: it is binarised,
optionally cropped and morphologically cleaned, split into connected
components, and the *N* largest vessels are labelled. All results are written
as `.vtk` files you can open in ParaView or PyVista.

---

## Repository layout

```
segment4Dflow/
├── train2d.py                 # Train a 2D Attention U-Net on 2D patches
├── train2d_selector.py        # Train a 2D "vessel selector" network (slice-wise)
├── train3d.py                 # Train a 3D U-Net / Attention U-Net on 3D patches  ← main training script
├── train3d_selector.py        # Train a 3D "vessel selector" network
├── train3d hierarchical.py    # Experimental: dual-path (local + global context) 3D U-Net
│
├── inference3d2c.py           # 3D, 2-channel inference on whole volumes  ← main inference script
├── segment3d2c.py             # 3D, 2-channel inference using overlapping patches
├── segment.py                 # Slice-wise (2D) segmentation of single-file VTK volumes
├── segment2Channels.py        # Slice-wise (2D) segmentation with 2 input channels
├── inference.py               # Legacy 2D aorta inference on HDF5 data + Dice evaluation
├── plot_dice.py               # Plot train/val/test Dice curves of a trained model
│
├── myFunctions/
│   └── data_handling.py       # Data loading, patching, augmentation, splitting, metrics,
│                              #   patch-wise inference, connected-component post-processing
├── medformer2D/               # 2D network zoo (UNet, UNet++, Attention U-Net, DA-UNet,
│                              #   TransUNet, Swin-UNet, MedFormer)
├── medformer3D/               # 3D network zoo (UNet, UNet++, Attention U-Net, VNet, UNETR,
│                              #   Swin-UNETR, VT-UNet, nnFormer, MedFormer, DualPathUNet)
│
├── environment.yml            # Conda environment ("torch")
└── piptorch.def               # Apptainer/Singularity container definition
```

### Network architectures

The `medformer2D/` and `medformer3D/` packages hold a set of medical
segmentation architectures, adapted from public implementations (MedFormer /
CBIM, milesial Pytorch-UNet, MONAI UNETR & Swin-UNETR, TransUNet, Swin-Unet,
nnFormer, VNet). The training scripts use these models:

- **`UNet`** (3D): `UNet(in_ch, base_ch, scale, kernel_size, num_classes=1, block='ConvNormAct', pool=True, norm='bn')`
- **`AttentionUNet`** (2D/3D): U-Net with attention gates on the skip connections
- **`DualPathUNet`** (3D, experimental): fuses a local-patch U-Net with a coarser global-context U-Net

### `myFunctions/data_handling.py`: main helpers

| Function | Purpose |
|----------|---------|
| `getData(img_path, channels, dims, patching, NP)` | Read every subject folder, stack the channels, normalise, and cut into 2D/3D patches of size `NP` |
| `getAllPatches`, `getAllPatches2Channels` | Older patch loaders (1 or 2 channels) |
| `patching2d`, `patching3d` | Split a volume into non-overlapping patches |
| `augment_and_expand_3d_patches` | Random 90° rotations, flips and Gaussian noise; adds `num_aug` copies per patch |
| `split4Train`, `split4Train2Channels`, `split4TrainAll` | Split into train / validation / test `Dataset`s (default 80 % train) |
| `normalize_and_standardize` | Intensity normalisation |
| `dice_score`, `create_mask1ch` | Dice metric; threshold probabilities to a binary mask (0.5) |
| `infer_3d_unet(model, volume, patch_size, overlap, mySigmoid)` | Sliding-window inference with overlap averaging over a large 3D volume |
| `getLargestSegmentations(mask, nr_masks)` | Label the `nr_masks` largest connected components |
| `list_directories`, `write_dict_to_file` | Utilities |

---

## Installation

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate torch
pip install pyvista einops medpy
```

`pyvista` (VTK I/O), `einops` (transformer models) and `medpy` (imported by
`segment*.py`) are not in `environment.yml`, so install them separately.

### Option B: Apptainer / Singularity (for GPU clusters)

```bash
apptainer build piptorch.sif piptorch.def
apptainer exec --nv piptorch.sif python train3d.py
```

### Device selection

Each script picks the device itself: Apple `mps` if it is available, then
`cuda`, then `cpu`. Note that `train3d.py` only builds the model in the `cuda`
branch, so run 3D training on a CUDA GPU.

---

## Data format

The training and 3D inference scripts expect **one folder per subject**. Each
folder holds one VTK image (`pyvista.ImageData`, array name `data`) per channel
and one for the ground-truth mask:

```
<img_path>/
├── <subject_A>/
│   ├── magnitude_<subject_A>.vtk
│   ├── cd_<subject_A>.vtk
│   └── segmentation_<subject_A>.vtk     # ground truth (or mask_.vtk as fallback)
├── <subject_B>/
│   └── ...
```

- Channel file names follow `<channel>_<subject>.vtk`, where `<channel>` is an
  entry of the `channels` list (e.g. `['magnitude', 'cd']`).
- The mask is read from `segmentation_<subject>.vtk`. If that file is missing,
  `mask_.vtk` is used. The scalar array may be named `scalars` or `data`.
- Test folders used for inference have the same layout. They need no mask.

The legacy `inference.py` reads HDF5 files that contain `magnitude` and `mask`
datasets instead.

---

## Usage

The scripts take **no command-line arguments**. You configure each one by
editing the variables at the top of the file. Most importantly, **replace the
hard-coded absolute paths** (`/Users/jonathan/...`, `/home/jnestmann/...`,
`/mnt/itetstor/...`) with your own:

- the `sys.path.append(...)` line that points to `medformer3D/`
- `w_dir`, `data_dir`, `img_path`, `test_path`, `model_path`, `out_path` / `res_path`

Run scripts from the repository root, so that `from myFunctions.data_handling import *`
resolves. `segment.py`, `segment2Channels.py` and `segment3d2c.py` import
`data_handling` directly, so put `myFunctions/` on the path for those:

```bash
# Linux / macOS
export PYTHONPATH="$PWD/myFunctions:$PWD/medformer3D:$PYTHONPATH"
# Windows PowerShell
$env:PYTHONPATH = "$PWD\myFunctions;$PWD\medformer3D"
```

### 1. Train a 3D segmentation model (`train3d.py`)

Key settings at the top of the script:

```python
model_type      = 'unet'      # 'unet' or 'attention'
n_patch         = 64          # cubic patch size
batch_numb      = 12
n_epochs        = 1000
n_input         = 2           # number of input channels
init_lr         = 1e-3
nr_base_filters = 64          # 16, 32, 64
num_aug         = 5           # augmented copies per patch
scheduler_flag  = 'cosine'    # CosineAnnealingLR (T_max=50)
start_epoch     = 0           # >0 resumes from a saved checkpoint
model_dir       = 'model53'   # output sub-folder name
channels        = ['magnitude', 'cd']
```

```bash
python train3d.py
```

The training pipeline:
1. loads all subjects, normalises them and cuts them into `n_patch³` patches (`getData`)
2. augments the patches (rotation, flip, noise)
3. splits them 80/20 into train/val/test
4. trains with MSE loss on the sigmoid output, using Adam (`betas=(0.5, 0.999)`)
5. computes the Dice score on validation every epoch, and on the test set every 10 epochs

Outputs go to `<w_dir>/output/<model_dir>/`:

| File | Content |
|------|---------|
| `SegmentBrainVessels_XXXXX.pt` | Checkpoint every 10 epochs (`epoch`, `model_state_dict`, `optimizer_state_dict`) |
| `parameters.txt` | Hyper-parameters of the run |
| `train_loss.npy`, `val_loss.npy`, `train_dice.npy`, `val_dice.npy`, `test_dice.npy`, `learning_rate.npy` | Training curves |
| `loss.png`, `dice_score.png`, `test_dice.png`, `learning_rate.png` | Plots, updated every epoch |

### 2. Run 3D inference (`inference3d2c.py`)

Set the trained model and test data:

```python
test_folder     = 'test_c2_all'
model_dir       = 'model51'
model_type      = 'unet'
ref_epoch       = 590         # checkpoint epoch to load
nr_base_filters = 32          # must match training
channels        = ['magnitude', 'cd']
crop_z          = 10          # zero out N slices at both z ends
use_mag         = False       # optionally suppress voxels below the magnitude noise level
nr_vessels      = 8           # number of largest vessels to label
nr_dilate, nr_erode = 0, 0    # morphological clean-up iterations
```

```bash
python inference3d2c.py
```

This script segments each whole volume in a single forward pass (it runs on
CPU by default). For volumes too large for memory, use **`segment3d2c.py`**
instead. It calls `infer_3d_unet` for sliding-window inference with
`n_patch`-sized patches and `n_overlap` voxels of overlap.

Outputs go to `<w_dir>/results/results_<model_dir>_ep<ref_epoch>/`, one set per subject:

| File | Content |
|------|---------|
| `segmentation_ep<E>_<subject>.vtk` | Binary vessel mask |
| `cc_ep<E>_<subject>.vtk` | Iso-surface (0.5) of the mask with a `RegionId` per connected component |
| `label_mask_ep<E>_<subject>.vtk` | Volume labelling the `nr_vessels` largest vessels |

Open these files in **ParaView** or load them with `pyvista.read(...)`.

### 3. 2D (slice-wise) workflow

- **Training:** `train2d.py` trains a 2D `AttentionUNet` on 2D patches of
  `magnitude` + `velocity` (`getAllPatches2Channels`). Checkpoints and example
  overlays (prediction in blue, ground truth in red) are saved to
  `output/<model_dir>/Results_2D/`.
- **Segmentation:** `segment.py` (single input VTK files in a flat folder) and
  `segment2Channels.py` (per-subject folders, 2 channels) run the 2D model
  slice by slice along z. They then apply the same post-processing and VTK
  export as the 3D scripts. `segment.py` also writes one file per vessel
  (`seg_ep<E>_id<i>_<subject>`).
- **Legacy aorta model:** `inference.py` evaluates a 2D aorta model
  (`Model_2D_mix/SegmentAorta_XXXXX.pt`) on HDF5 data. It keeps the largest
  connected component per slice and reports the per-subject and overall Dice.

### 4. Vessel selector networks

`train2d_selector.py` and `train3d_selector.py` train a second network
(`SelectBrainVessels`). Its input is an existing vessel-network segmentation
(`segmentation` / `network_seg`) and its target is a mask of the vessels of
interest. The network learns to **pick the relevant vessels** out of a full
segmentation.

### 5. Plot training curves

Set `w_dir` and `model_dir` in `plot_dice.py`, then run:

```bash
python plot_dice.py
```

The script writes `dice_score_scaled.png` (train/val/test Dice curves) to the model folder.

---

## Notes and known limitations

- **Hard-coded paths:** every script contains absolute paths from the
  original development machines, which you must adapt (see Usage).
- **Configuration in code:** there is no CLI or config file. Hyper-parameters
  are variables at the top of each script.
- **Architecture must match the checkpoint:** at inference, `model_type`,
  `nr_base_filters`, `scale`, `kernel_size` and `n_input` must equal the values
  used in training. Check `parameters.txt` in the model folder.
- `train2d.py`, `segment*.py` set `device = 'gpu'` instead of `'cuda'`, so on a
  CUDA machine without MPS they need that string corrected.
- `train3d hierarchical.py` (dual-path model) is **work in progress**: its
  data loader (`getPatches`) and training loop are not finished yet.
- Seeds are fixed (`random`, `numpy`, `torch` = 10) for reproducibility.
- `__pycache__/` and `.DS_Store` are committed and can be ignored.

## Author

Jonathan Nestmann
