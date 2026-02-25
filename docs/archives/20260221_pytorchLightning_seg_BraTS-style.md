Here's a robust, modular Python script that recursively traverses the BraTS directory structure, loads NIfTI files for training and validation (handling both structural and segmentation data), and sets up a typical training/validation pipeline using PyTorch + MONAI.

> ✅ **Features**:
> - Recursively finds all `.nii.gz` files (supports both `.nii` and `.nii.gz`)
> - Handles the 4-modalities (`t1`, `t1ce`, `t2`, `flair`) and segmentation (`seg`)
> - Aligns modalities and segmentation by case (uses same folder name)
> - Preprocesses with MONAI transforms (resize, normalize, crop/pad, etc.)
> - Uses `DataLoader` with multi-worker support
> - Configurable training/validation split (or use official `training/` and `validation/`)
> - Includes skeleton for model training (e.g., with `UNet`, Dice loss)

---

### ✅ Prerequisites
```bash
pip install torch torchvision torchaudio
pip install monai[nibabel,libnvtx]
pip install scikit-learn  # optional, if doing custom split
```

---

### 📁 Directory Assumptions (BraTS format)
```
Brats2020/
├── training/                # Has both images & segmentation
│   ├── BRATS_001/
│   │   ├── BRATS_001_t1.nii.gz
│   │   ├── BRATS_001_t1ce.nii.gz
│   │   ├── BRATS_001_t2.nii.gz
│   │   ├── BRATS_001_flair.nii.gz
│   │   └── BRATS_001_seg.nii.gz   ← segmentation
│   ├── BRATS_002/ ...
│   └── ...
├── validation/              # Only 4 modalities (no seg)
│   ├── BRATS_001/
│   │   ├── BRATS_001_t1.nii.gz
│   │   ├── ...
│   │   └── BRATS_001_flair.nii.gz
│   └── ...
└── ...
```

---

### ✅ Full Script: `brats_dataset_and_train.py`

```python
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd,
    ResizeWithPadOrCropd, Compose, ToTensord, EnsureTyped, Spacingd
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import compute_dice
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import re

# -------------------------------
# 1. Utility: Recursive glob for BraTS-style paths
# -------------------------------
def find_brats_cases(root_dir: str) -> List[Dict[str, str]]:
    '''
    Recursively find all BraTS cases (t1, t1ce, t2, flair, seg).
    Returns list of dicts: {t1: path, t1ce: path, ..., seg: path or None}
    '''
    root = Path(root_dir)
    training_dir = root / 'training'
    val_dir = root / 'validation'

    cases = []

    def process_dir(path):
        if not path.exists():
            return
        # Find subdirectories (case folders)
        for case_folder in sorted(path.glob('BRATS_*')):
            if not case_folder.is_dir():
                continue

            case_data = {'case_name': case_folder.name, 'modality_paths': {}}
            # Look for modality files
            for mod in ['t1', 't1ce', 't2', 'flair', 'seg']:
                pattern = case_folder / f'{case_folder.name}_{mod}.nii*'
                files = list(glob.glob(str(pattern)))  # supports .nii or .nii.gz
                if files:
                    case_data['modality_paths'][mod] = files[0]
                elif mod == 'seg':
                    # For val set, seg should be missing → we mark it None explicitly
                    case_data['modality_paths'][mod] = None
                else:
                    raise FileNotFoundError(f'Missing modality '{mod}' in {case_folder}')
            cases.append(case_data)

    if training_dir.exists():
        process_dir(training_dir)
    if val_dir.exists():
        process_dir(val_dir)

    # Optional: Validate each case has at least 4 modalities
    cases = [c for c in cases if len(c['modality_paths']) >= 4]
    print(f'Found {len(cases)} cases with full modalities across {root_dir}')
    return cases


# -------------------------------
# 2. Custom Dataset
# -------------------------------
class BraTSDataset(Dataset):
    def __init__(
        self,
        cases: List[Dict[str, str]],
        transform: Compose,
        train_mode: bool = True,  # if True, expects 'seg' to be present
        image_size: Tuple[int, int, int] = (128, 128, 128),
    ):
        self.cases = cases
        self.transform = transform
        self.train_mode = train_mode

        # Validate: if train_mode=True, all cases must have seg
        if train_mode and not all(c['modality_paths']['seg'] is not None for c in cases):
            raise ValueError('Training dataset requires segmentation labels for all cases.')

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        inputs = [case['modality_paths'][mod] for mod in ['t1', 't1ce', 't2', 'flair']]
        target_path = case['modality_paths']['seg'] if self.train_mode else None

        data_dict = {
            't1': inputs[0],
            't1ce': inputs[1],
            't2': inputs[2],
            'flair': inputs[3],
        }
        if self.train_mode and target_path:
            data_dict['seg'] = target_path

        # Apply transforms ( LoadImaged → EnsureChannelFirstd → ... )
        # Note: transform expects filenames in keys
        if self.transform:
            data_dict = self.transform(data_dict)

        # Combine modalities into single tensor (C, H, W, D)
        x = torch.stack([
            data_dict['t1'],
            data_dict['t1ce'],
            data_dict['t2'],
            data_dict['flair'],
        ], dim=0)  # shape: [4, H, W, D]

        y = None
        if self.train_mode:
            # Segmentation: combine ET, TC, WT labels into 3-class (or 1 for whole tumor)
            # BraTS 2020+ uses multi-class (0, 1, 2, 4) → we usually want ET (4), WT (1+2+4), TC (1+4)
            # Simplify to 2-class (0=background, 1=tumor) or keep 3-class if desired
            y = data_dict['seg']  # shape: [1, H, W, D]
            # Optional: post-process seg (e.g., relabel for 3-class)
            # Example: whole tumor = label 1 (0,1,2,4 → 0 vs ≥1)
            # y = (y > 0).float()

        return {'image': x, 'label': y}


# -------------------------------
# 3. Data & Transforms Setup
# -------------------------------
def get_transforms(image_size=(128, 128, 128), is_train=True):
    # Common intensity scaling range: BraTS is already preprocessed
    # Usually values in [0, 1000], so we clip & scale to [0,1]
    intensity_range = (0, 1000)

    transforms_list = [
        LoadImaged(keys=['t1', 't1ce', 't2', 'flair'] + (['seg'] if is_train else [])),
        EnsureChannelFirstd(keys=['t1', 't1ce', 't2', 'flair'] + (['seg'] if is_train else [])),
        # Optional: Spacing to isotropic (e.g., [1,1,1]) — useful if raw data varies
        # Spacingd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest')),
        ScaleIntensityRanged(
            keys=['t1', 't1ce', 't2', 'flair'],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Crop to brain region: foreground-based cropping
        CropForegroundd(
            keys=['t1', 't1ce', 't2', 'flair'] + (['seg'] if is_train else []),
            source_key='t1',  # crop based on T1
        ),
        # Pad/Crop to fixed size
        ResizeWithPadOrCropd(
            keys=['t1', 't1ce', 't2', 'flair'] + (['seg'] if is_train else []),
            spatial_size=image_size,
        ),
        ToTensord(keys=['t1', 't1ce', 't2', 'flair'] + (['seg'] if is_train else [])),
    ]

    if is_train:
        # Add data augmentation (only for training)
        from monai.transforms import RandFlipd, RandRotate90d, RandShiftIntensityd
        transforms_list.extend([
            RandFlipd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], spatial_axis=0, prob=0.5),
            RandFlipd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=['t1', 't1ce', 't2', 'flair', 'seg'], prob=0.5),
            RandShiftIntensityd(keys=['t1', 't1ce', 't2', 'flair'], offsets=0.1, prob=0.5),
        ])

    return Compose(transforms_list)


# -------------------------------
# 4. Training & Validation Loop
# -------------------------------
def main():
    root_dir = '/path/to/BraTS/data'  # ✅ UPDATE THIS!
    image_size = (128, 128, 128)
    batch_size = 2
    num_workers = 4

    # Get all cases (training + val)
    all_cases = find_brats_cases(root_dir)

    # Split into train/val sets (or use official splits)
    # Option A: Use official folders
    training_cases = [c for c in all_cases if 'training' in c.get('modality_paths', {}).get('t1', '')]
    validation_cases = [c for c in all_cases if 'validation' in c.get('modality_paths', {}).get('t1', '')]

    # Option B: Use all training + manually split (e.g., 90/10)
    # from sklearn.model_selection import train_test_split
    # train_cases, val_cases = train_test_split(all_cases, test_size=0.1, random_state=42)

    print(f'Training cases: {len(training_cases)}')
    print(f'Validation cases: {len(validation_cases)}')

    train_ds = BraTSDataset(
        training_cases,
        transform=get_transforms(image_size=image_size, is_train=True),
        train_mode=True,
    )
    val_ds = BraTSDataset(
        validation_cases,
        transform=get_transforms(image_size=image_size, is_train=False),
        train_mode=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------
    # Model Setup
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        spatial_dims=3,
        in_channels=4,  # 4 modalities
        out_channels=1,  # binary segmentation (or 3 for multi-class)
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_fn = DiceLoss(sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # Training Loop
    # -------------------------
    epochs = 10
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        model.train()
        epoch_loss = 0

        for batch_data in train_loader:
            inputs = batch_data['image'].to(device)  # [B, 4, H, W, D]
            labels = batch_data['label'].to(device)  # [B, 1, H, W, D]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Train Loss: {avg_loss:.4f}')

        # Validation (optional: compute Dice metric)
        model.eval()
        val_scores = []
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data['image'].to(device)
                labels = batch_data['label'].to(device)
                outputs = model(inputs)
                # Compute Dice for thresholded outputs
                dice = compute_dice(y_pred=(outputs > 0.5).float(), y=labels, include_background=False).mean().item()
                val_scores.append(dice)
        print(f'Val Dice: {np.mean(val_scores):.4f}')

    # Save model
    torch.save(model.state_dict(), 'brats_unet_final.pth')
    print('Model saved!')


if __name__ == '__main__':
    main()
```

---

### 📝 Notes & Customization Tips

1. **Modality order matters**: We consistently use `['t1', 't1ce', 't2', 'flair']`.
2. **Label handling**:
   - BraTS labels: `0=background`, `1= Necrotic/Non-enhancing`, `2= Edema`, `4= Enhancing`.
   - For binary (whole tumor): `label > 0`
   - For 3-class (ET, WT, TC): see [BraTS label conventions](https://www.med.upenn.edu/cbica/brats2020/task-details.html).
   - You can add preprocessing after loading seg, e.g.:
     ```python
     # Inside BraTSDataset.__getitem__ after transform:
     seg = data_dict['seg']
     # Convert multi-class to whole tumor: 0 vs ≥1
     data_dict['seg'] = (seg > 0).float()
     ```
3. **Spacing**: Uncomment `Spacingd` if your data has non-isotropic voxels.
4. **Performance**: Increase `num_workers`, use `persistent_workers=True`, or `prefetch_factor`.
5. **Validation set labels**: In BraTS validation, segmentation is withheld — you can optionally submit to [Synapse](https://www.synapse.org/brats) for evaluation.
6. **Post-op files**: If you have `post-op` folders, modify `find_brats_cases()` to include those (they likely lack `seg`, so treat as validation-only).

---

Let me know if you need:
- Multi-class segmentation (ET/WT/TC)
- 3D patches (sliding window inference)
- Inference on raw validation data (no labels)
- Integration withWeights & Biases / TensorBoard

---
# Version 2 - Transfer to Post-Operative DataSet
Great question! Transferring knowledge from a pre-trained model (trained on BraTS training + validation data) to **post-operative MRI datasets** requires careful handling because:

### 🧠 Key Challenges with Post-Op Data:
| Issue | Explanation |
|-------|-------------|
| **Altered anatomy** | Resection cavities, surgical implants, edema, shifts → differs significantly from native tumor anatomy |
| **Missing labels** | No ground-truth segmentation (typically) → unsupervised/self-supervised domain adaptation needed |
| **Modality contrast differences** | Post-op scans may use different protocols, contrast timing, or have artifacts |
| **Domain gap** | Model trained on *de novo* gliomas may struggle with post-op changes |

But with the right strategy, you *can* leverage the pre-trained BraTS model — here’s how:

---

## ✅ 4 Practical Strategies (with Code + Implementation Guide)

---

### 🧩 Strategy 1: **Fine-Tuning on Limited Post-Op Labels (Supervised / Semi-Supervised)**
Use a small set of expert-annotated post-op cases.

#### ✅ Steps:
1. **Pre-train** on BraTS (you already have `brats_unet_final.pth`)
2. **Fine-tune** on your post-op dataset (with manual segmentations)
3. Use **learning rate ramp-up**, **label smoothing**, or **contrastive regularization** to prevent catastrophic forgetting.

#### 🔧 Code Additions:
```python
# Load pre-trained weights
pretrained_path = 'brats_unet_final.pth'
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# Load weights (skip mismatched layers if needed — not needed here)
model.load_state_dict(torch.load(pretrained_path, map_location=device))

# Set up post-op dataset
postop_cases = load_postop_nifti_cases(root_dir_postop)  # ✅ custom function
postop_ds = BraTSDataset(
    postop_cases,
    transform=get_transforms(image_size=(128,128,128), is_train=True),
    train_mode=True,
)

postop_loader = DataLoader(postop_ds, batch_size=2, shuffle=True, num_workers=4)

# Use lower LR + freeze early layers optionally
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

# Optional: freeze first 2 blocks
# for param in list(model.encoder.layers)[:4]: param.requires_grad = False

# Train for fewer epochs (e.g., 5–10)
for epoch in range(10):
    model.train()
    for batch in postop_loader:
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        # ... same training loop ...
```

> 💡 **Tip**: Use `LabelSmoothingDiceLoss` or `TverskyLoss` (better for imbalanced surgical cavities).

---

### 🔄 Strategy 2: **Domain Adaptation (Unsupervised / Self-Supervised)**
If you have *no labels* for post-op data.

#### Options:
1. **Test-Time Adaptation (TTA)**  
2. **CycleGAN/DiscoGAN** to translate post-op → pre-op-like scans  
3. **Self-supervised pretraining on post-op data** (e.g., jigsaw, rotation prediction)

#### 🔧 Example: **TTA with Entropy Minimization**
```python
# After training (on BraTS), run on post-op *without labels*
model.eval()
with torch.no_grad():
    for case_data in test_loader:
        inputs = case_data['image'].to(device)  # 4 modalities
        # Forward pass
        pred_logits = model(inputs)
        pred_prob = torch.sigmoid(pred_logits)

        # Entropy minimization loss (self-supervised refinement)
        eps = 1e-6
        entropy = -pred_prob * torch.log(pred_prob + eps) - (1 - pred_prob) * torch.log(1 - pred_prob + eps)
        tta_loss = entropy.mean()

        # Optional: backprop *only* BN statistics (e.g., using AdaIN / EvoNorm)
        # or update via gradient-based TTA (e.g., MEMO, RTA)
```

> 🔎 Research options:
> - [**AdaIn + Domain Discriminator**](https://arxiv.org/abs/1802.07948)  
> - [**MixMatch for semi-supervised domain adaptation**](https://arxiv.org/abs/1905.02244)  
> - **Feature alignment**: minimize MMD between BraTS & post-op features

---

### 🔎 Strategy 3: **Multi-Modality Knowledge Transfer via Modality Agnostic Features**
BraTS models learn robust *structural* features (e.g., tumor boundaries, edema patterns) — but post-op cavities ≠ tumors.

#### ✅ Fix: Use **intermediate features** as priors

```python
# Extract encoder features from BraTS model
class FeatureExtractor(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.encoder = unet.encoder
        self.down_samples = unet.down_samples
        self.bottleneck = unet.bottleneck

    def forward(self, x):
        features = []
        x = self.encoder[0](x)
        features.append(x)
        for down, layer in zip(self.down_samples, self.encoder[1:]):
            x = down(x)
            x = layer(x)
            features.append(x)
        return features  # [B, C_i, H_i, W_i, D_i]

# Use features as conditionings (e.g., via cross-attention or skip connections)
```

→ Train a *lightweight decoder* on post-op data using BraTS encoder weights.

---

### 🧪 Strategy 4: **Post-Op Specific Post-Processing**
Leverage BraTS model’s predictions + *heuristic rules* for surgical artifacts:

| Issue | Fix |
|-------|-----|
| **Resection cavity vs tumor** | Use intensity + shape priors: cavity = hyperintense in T1w (post-contrast), irregular, contiguous to skull |
| **Implant artifacts** | Mask regions with extreme intensity outliers (`ClipIntensityPercentilesd`) |
| **Shifted anatomy** | Register pre-op (if available) → post-op via Elastix / ANTs |

#### Example (post-processing mask):
```python
import numpy as np
from scipy import ndimage

def postprocess_cavity_mask(pred, t1ce):
    '''Refine BraTS prediction for surgical cavity'''
    pred = pred.cpu().numpy()[0,0] > 0.5
    t1ce_np = t1ce.cpu().numpy()[0,0]

    # 1. Keep only hyperintense voxels in T1ce (common for post-op enhancing rim/cavity)
    cavity_mask = pred & (t1ce_np > np.percentile(t1ce_np, 90))

    # 2. Remove disconnected components < 500 mm³ (likely noise)
    labeled, _ = ndimage.label(cavity_mask)
    sizes = np.bincount(labeled.ravel())
    cavity_mask = labeled == np.argmax(sizes[1:])+1  # keep largest
    cavity_mask = ndimage.binary_fill_holes(cavity_mask)

    return torch.from_numpy(cavity_mask).float()
```

---

## 🚀 Practical Checklist for Transfer

| ✅ Step | Action |
|--------|--------|
| **1. Verify modality alignment** | Ensure post-op has T1, T1ce, T2, FLAIR — same order as training |
| **2. Re-center on brain** | Use `CropForegroundd(source_key='t1')` — post-op may have large voids |
| **3. Normalize carefully** | BraTS intensity range ≠ post-op: compute `intensity_range = np.percentile(t1, [1, 99])` per case |
| **4. Start fine-tuning** | Freeze early layers, train decoder + bottleneck first |
| **5. Validate with clinical experts** | Surgical cavity ≠ tumor — semantics differ! |

---

## 📌 Bonus: Dataset for Post-Op Glioma Segmentation (Public)
- **NuSeg** (Neurosurgery Segmentation Challenge): [https://nuSEG.grand-challenge.org/](https://nuSEG.grand-challenge.org/)
- **Surgical Brain Tumor Segmentation (SBTS)**: [https://sbts.grand-challenge.org/](https://sbts.grand-challenge.org/)
- Use their models or data to initialize your domain adaptation.

---

### Let me know if you'd like:
- A full `find_postop_cases()` loader
- Code for **TTA with entropy regularization**
- Integration with **ANTsPy** for registration-aware preprocessing
- A UNet variant that **ignores cavity artifacts** (e.g., attention gates)

Happy to tailor this to your exact post-op setup! 🧠✨