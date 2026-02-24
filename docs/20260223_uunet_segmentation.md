Here’s a **step-by-step pipeline** to perform brain tumor segmentation on *post-treatment* NIfTI data (no baseline required) using a **pretrained nnUNet model**, with PyTorch-based inference, patient/timepoint organization, and visualization.

Since there is **no pre-treatment baseline**, we focus on segmentation of each post-treatment scan individually — ideal for monitoring recurrence/progression.

---

### ✅ Assumptions & Data Structure

Your data directory (e.g., `./data/post_treatment/`) is structured as:

```
data/
└── patient001/
    ├── baseline/          # optional: may be empty or contain pre-op scan for context (ignored in this pipeline)
    ├── post_op_2days/
    │   ├── patient001_post_op_2days_t1c.nii.gz
    │   ├── patient001_post_op_2days_t1n.nii.gz
    │   ├── patient001_post_op_2days_t2f.nii.gz
    │   ├── patient001_post_op_2days_t2w.nii.gz
    │   └── (optional) patient001_post_op_2days_tumormask.nii.gz  # ground truth (if available)
    └── followup_3m/
        ├── ...
```

> ✅ Modalities: **t1c, t1n, t2f, t2w**  
> ✅ Target: **Whole tumor, enhancing tumor, necrosis, etc.** (depends on trained model — typically BraTS-style labels 1–4)  
> ✅ No need for a 'pre-treatment' reference — each scan is processed independently.

---

## 🔧 Step-by-Step Pipeline

### **Step 1: Install Required Packages**

```bash
# Create & activate env (optional)
conda create -n nnunet python=3.10 -y
conda activate nnunet

# Install PyTorch (GPU recommended)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install nnUNet (v2 recommended, but v1 also works)
pip install nnunetv2
pip install numpy SimpleITK nibabel matplotlib scikit-image tqdm pandas

# Optional: for interactive visualization
pip install ipywidgets plotly
```

> 🔔 **nnUNet Note**: Use [nnUNetV2](https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2) (more accurate, faster) if available. Pretrained BraTS models (e.g., `nnUNetTrainer_5fold`, `nnUNetTrainer_5fold_cv`) are ideal.

---

### **Step 2: Download & Setup Pretrained Model**

For BraTS-style segmentation (WT, TC, ET, or 4-class), download a pretrained model:

#### Option A: Use nnUNet’s official BraTS models (recommended)

Download from [nnUNet Model Zoo](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/model_zoo/plan_and_transfer_training.md) — e.g., for BraTS 2021/2023:

```bash
# Example: nnUNetv2 trained on BraTS 2021 (4-class, 5-fold)
# Download nnUNetTrainer__nnUNetPlans__3d_fullres model
# and its folds (0–4)

# You’ll get something like:
# models/
#   └── nnUNet/
#       └── 3d_fullres/
#           └── nnUNetTrainer__nnUNetPlans__3d_fullres/
#               ├── dataset_fingerprint.json
#               ├── plans.json
#               └── fold_0/
#                   └── model.final_checkpoint
#               └── fold_1/ ... etc.
```

Or use `nnunetv2` CLI:

```bash
nnunetv2 download_pretrained_model 3d_fullres nnUNetTrainer__nnUNetPlans__3d_fullres
```

> 📌 Ensure the model was trained on **same modality schema** (T1n, T1c, T2w, T2f). If using BraTS models, labels may be:  
> `0: background`, `1: necrosis`, `2: non-enhancing tumor`, `3: enhancing tumor`, `4: whole tumor (merged)` — *check `plans.json`*.

#### Option B: Train your own (if you have >30 cases & need custom post-treatment model)
- Skip for now unless necessary.

---

### **Step 3: Create Inference Script**

Create `inference_post_treatment.py`:

```python
import os
import sys
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List, Tuple
import torch
import SimpleITK as sitk

from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.network_initialization import restore_model
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

# -------------------------------
# Configuration
# -------------------------------
DATASET_ROOT = Path('./data/post_treatment')
OUTPUT_DIR = Path('./predictions')
PRETRAINED_MODEL_DIR = Path('./models/nnUNet/3d_fullres/nnUNetTrainer__nnUNetPlans__3d_fullres')

# Modalities to use (order must match training!)
MODALITY_ORDER = ['t1n', 't1c', 't2w', 't2f']  # nnUNet expects this order for BraTS-style
# NOTE: t1n = T1-weighted, t1c = T1-weighted contrast-enhanced, t2w = T2-weighted, t2f = T2-FLAIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OVERLAP = 0.5
SAVE_PROBS = False  # set True if you want soft predictions

# -------------------------------
# Utility Functions
# -------------------------------

def find_patient_scans(data_dir: Path) -> List[Tuple[str, str, List[Path]]]:
    '''
    Returns list of (patient_id, timepoint, list_of_modality_paths)
    for scans with all required modalities.
    '''
    patients = []
    for patient_dir in sorted(data_dir.glob('patient*')):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        for timepoint_dir in sorted(patient_dir.glob('*')):
            if not timepoint_dir.is_dir():
                continue
            timepoint = timepoint_dir.name

            modality_paths = {}
            for mod in MODALITY_ORDER:
                # Case-insensitive glob
                candidates = list(timepoint_dir.glob(f'**/*_{mod}.nii*'))
                if not candidates:
                    print(f'⚠️ Missing modality '{mod}' in {patient_id}/{timepoint}')
                    break
                modality_paths[mod] = candidates[0]  # assume single match

            if len(modality_paths) == len(MODALITY_ORDER):
                scan_paths = [modality_paths[mod] for mod in MODALITY_ORDER]
                patients.append((patient_id, timepoint, scan_paths))
    return patients

def preprocess_for_nnunet(modality_paths: List[Path], output_dir: Path) -> str:
    '''
    Copy/convert modality NIfTis to nnUNet raw format (simulated).
    nnUNetv2 inference expects:
        - Folder with input images (named 'img0000.nii.gz', etc.)
        - plans.json and model checkpoint in PRETRAINED_MODEL_DIR

    We create a temp folder with renamed images in correct modality order.
    '''
    tmp_folder = output_dir / 'temp_inference'
    tmp_folder.mkdir(exist_ok=True, parents=True)

    input_images = []
    for i, mod_path in enumerate(modality_paths):
        out_name = tmp_folder / f'img{i:04d}.nii.gz'
        # Simple copy — nnUNet handles normalization internally
        # (if you need bias-field correction or skull-stripping, do it here)
        sitk.WriteImage(sitk.ReadImage(str(mod_path)), str(out_name))
        input_images.append(str(out_name))

    return str(tmp_folder)

def run_nnunet_inference(input_folder: str, output_folder: str):
    '''
    Run nnUNet inference via official API.
    '''
    # Get plans & config
    plans_manager = PlansManager(PRETRAINED_MODEL_DIR / 'plans.json')
    configuration_manager = plans_manager.get_configuration('3d_fullres')

    # Convert input to list of file paths (nnUNetv2 expects this format)
    input_files = sorted([str(f) for f in Path(input_folder).glob('img*.nii.gz')])
    if len(input_files) != len(MODALITY_ORDER):
        raise ValueError(f'Expected {len(MODALITY_ORDER)} modalities, got {len(input_files)}')

    # Run prediction
    predict_from_raw_data(
        list_of_lists_or_source_folder=[input_files],
        output_file_or_folder=output_folder,
        model_training_or_checkpoint_folder=str(PRETRAINED_MODEL_DIR),
        fold=[0, 1, 2, 3, 4],  # use all folds for ensemble
        step_size=OVERLAP,
        use_gaussian=True,
        use_tta=False,  # disable TTA for speed
        mixed_precision=True,
        verbose=True,
        save_probs=SAVE_PROBS
    )

def postprocess_labels(mask_nii: nib.Nifti1Image) -> np.ndarray:
    '''
    Optional: clean mask (remove small components, etc.)
    Returns label array (same shape as input).
    '''
    import scipy.ndimage as ndi
    arr = mask_nii.get_fdata()
    # Example: remove small connected components (e.g., < 100 voxels)
    label_arr, _ = ndi.label(arr > 0)
    sizes = np.bincount(label_arr.ravel())
    mask_sizes = sizes > 100
    mask_sizes[0] = 0  # don’t keep background
    cleaned = mask_sizes[label_arr]
    return cleaned.astype(np.uint8)

def visualize_prediction(
    patient_id: str,
    timepoint: str,
    modalities: List[Path],
    pred_path: Path,
    output_dir: Path,
    slices_to_show: List[int] = None
):
    '''
    Visualize overlay of T1c + prediction (and T2f if helpful).
    Uses matplotlib.
    '''
    import matplotlib.pyplot as plt

    pred_nii = nib.load(pred_path)
    pred = pred_nii.get_fdata()
    # If probabilistic, argmax to get label map
    if pred.ndim == 4:  # [H,W,D,classes]
        pred = np.argmax(pred, axis=-1).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    # T1c (modality 1), T2f (modality 3) are most diagnostically useful
    t1c_nii = nib.load(modalities[1])
    t2f_nii = nib.load(modalities[3])
    t1c = t1c_nii.get_fdata()
    t2f = t2f_nii.get_fdata()

    # Slices: center slice +上下 10%
    if not slices_to_show:
        slices_to_show = [t1c.shape[2]//2, int(t1c.shape[2]*0.3), int(t1c.shape[2]*0.7)]

    for i, sl in enumerate(slices_to_show):
        # T1c axial slice
        axes[2*i].imshow(t1c[:, :, sl].T, cmap='gray')
        overlay = np.ma.masked_where(pred[:, :, sl].T == 0, pred[:, :, sl].T)
        axes[2*i].imshow(overlay, cmap='jet', alpha=0.5)
        axes[2*i].set_title(f'T1c + Pred (slice {sl})')
        axes[2*i].axis('off')

        # T2f axial slice
        axes[2*i+1].imshow(t2f[:, :, sl].T, cmap='gray')
        axes[2*i+1].imshow(overlay, cmap='jet', alpha=0.5)
        axes[2*i+1].set_title(f'T2f + Pred (slice {sl})')
        axes[2*i+1].axis('off')

    plt.suptitle(f'{patient_id} — {timepoint}')
    plt.tight_layout()
    viz_path = output_dir / 'visualizations' / f'{patient_id}_{timepoint}_viz.png'
    viz_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(viz_path, dpi=150)
    plt.close()
    print(f'✅ Visualization saved to {viz_path}')

# -------------------------------
# Main Inference Loop
# -------------------------------
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or ''

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    patients = find_patient_scans(DATASET_ROOT)
    if not patients:
        print('❌ No valid patients found with all modalities!')
        sys.exit(1)

    print(f'➡️ Found {len(patients)} patients with full modality sets.')
    print(f'Using device: {DEVICE}\n')

    for patient_id, timepoint, mod_paths in patients:
        print(f'🔧 Processing: {patient_id} — {timepoint}')

        # 1. Prepare input (nnUNet format)
        input_folder = preprocess_for_nnunet(mod_paths, OUTPUT_DIR)

        # 2. Run inference
        output_folder = OUTPUT_DIR / 'raw_predictions' / patient_id / timepoint
        output_folder.mkdir(parents=True, exist_ok=True)

        run_nnunet_inference(input_folder, str(output_folder))

        # 3. Find output mask
        pred_files = list(output_folder.glob('*.nii.gz'))
        if not pred_files:
            print(f'❌ No prediction for {patient_id}/{timepoint}')
            continue

        pred_file = pred_files[0]  # usually one output file per case

        # 4. Optional post-processing & save
        pred_nii = nib.load(pred_file)
        pred_arr = pred_nii.get_fdata()

        # Optional: post-process (e.g., keep largest connected component)
        # (only if you want to enforce anatomical plausibility)
        # pred_arr = postprocess_labels(pred_nii)  # ⚠️ only works for binary masks!
        # For multi-class, use scipy.ndimage.label per class

        # Re-save (if modified)
        final_pred_path = OUTPUT_DIR / 'final_predictions' / patient_id / f'{patient_id}_{timepoint}_seg.nii.gz'
        final_pred_path.parent.mkdir(exist_ok=True, parents=True)
        nib.save(nib.Nifti1Image(pred_arr, pred_nii.affine, pred_nii.header), final_pred_path)

        # 5. Visualization
        visualize_prediction(patient_id, timepoint, mod_paths, final_pred_path, OUTPUT_DIR)

        # 6. Cleanup temp folder
        import shutil
        shutil.rmtree(OUTPUT_DIR / 'temp_inference', ignore_errors=True)

    print('\n✅ All done! Results saved to:')
    print(f'   - Raw predictions: {OUTPUT_DIR / 'raw_predictions'}/')
    print(f'   - Final (post-processed) masks: {OUTPUT_DIR / 'final_predictions'}/')
    print(f'   - Visualizations: {OUTPUT_DIR / 'visualizations'}/')
```

---

### ✅ Key Features & Notes

| Feature | Details |
|--------|---------|
| **No baseline needed** | Each timepoint is processed independently — ideal for longitudinal monitoring. |
| **nnUNet integration** | Uses official `nnunetv2` inference API for correct preprocessing & ensemble. |
| **Modality order** | Matches BraTS (`t1n`, `t1c`, `t2w`, `t2f`) — critical for correct segmentation. |
| **Visualization** | Shows T1c+T2f with segmentation overlay (axial slices). |
| **Post-processing** | Extensible: add CC cleanup, edge smoothing, or label remapping. |
| **Batch efficiency** | Loops over patients/timepoints; can parallelize with `multiprocessing`. |

---

### 📊 How to Interpret Outputs

nnUNet (BraTS-trained) typically outputs:
- **4-class labels**:
  - `0`: Background  
  - `1`: Necrotic/non-enhancing tumor  
  - `2`: Non-enhancing tumor (core)  
  - `3`: Enhancing tumor  
  - *(or sometimes merged labels)*

Check the model's `plans.json` → `label_dict` to confirm:

```python
import json
print(json.load(open(PRETRAINED_MODEL_DIR / 'plans.json'))['label_dict'])
```

> 💡 Tip: If you want to convert to BraTS challenge format (ET, TC, WT), use:  
> `ET = label==3`, `TC = label in [1,3]`, `WT = label in [1,2,3]`.

---

### 📁 Model Checkpoint Location (`PRETRAINED_MODEL_DIR`)

In your code, `PRETRAINED_MODEL_DIR` points to the **nnUNetv2 model directory**. This folder contains:

```
PRETRAINED_MODEL_DIR/
├── plans.json               # Configuration (modalities, splits, etc.)
├── dataset.json             # Dataset info (label names, modalities)
├── fold_0/
│   └── checkpoint_final.pth    # trained model weights
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/
```

✅ **What you need to do:**

1. **Download or train a model**  
   - For BraTS, use the official [nnUNetv2 BraTS23 model](https://github.com/MIC-DKFZ/nnUNet/blob/main/documentation/running_inference.md#inference-with-pretrained-models), which resides in:  
     ```
     nnUNet_results/nnUNetDataPlanes/PascalV2/nnUNetTrainer__nnUNetPlans__3d_fullres/  
     ```
     where `PascalV2` is the dataset ID (`Dataset201_BraTS23` in nnUNetv2).

2. **Set the environment variable**  
   ```python
   import os
   os.environ['nnUNet_results'] = '/path/to/nnUNet_results'
   PRETRAINED_MODEL_DIR = Path(os.environ['nnUNet_results']) / 'nnUNetDataPlanes' / 'PascalV2' / 'nnUNetTrainer__nnUNetPlans__3d_fullres'
   ```

   Or hardcode:
   ```python
   PRETRAINED_MODEL_DIR = Path('/path/to/your/model/folder')
   ```

3. **Ensure `dataset.json` and `plans.json` exist**  
   These define modalities (e.g. `['t1n', 't1c', 't2w', 't2f']`), so `MODALITY_ORDER` must match.

✅ That’s it! Once `PRETRAINED_MODEL_DIR` points to the correct folder with weights and configs, your pipeline will run.

---

Let me know if you need help:
- 📥 Downloading the BraTS model,
- 🔁 Adapting to custom datasets,
- 🧠 Visualizing per-label regions (ET/TC/WT), or  
- ⚙️ Optimizing inference (e.g., TTA, ensemble variants).

Your code is well-structured — great job! 🙌