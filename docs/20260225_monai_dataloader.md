Below is a **self‑contained Python script** that builds two MONAI `DataLoader`s – one for training (images + segmentation masks) and one for validation (images only).  
It assumes the directory layout you described and uses **dictionary‑based transforms** so the output can be fed directly to a `SegResNet` model.

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
MONAI data pipeline for a brain‑MRI segmentation task.

Structure expected under `dataset_root/`:

dataset_root/
├── train/
│   ├── patient001/
│   │   ├── patient001_t1.nii.gz
│   │   ├── patient001_t1ce.nii.gz
│   │   ├── patient001_t2.nii.gz
│   │   ├── patient001_flair.nii.gz
│   │   └── patient001_seg.nii.gz
│   └── patient002/ …
└── val/
    ├── patient050/ …
    └── …
'''

import os
from glob import glob
from typing import List, Tuple, Dict

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Spacingd,
    Orientationd,
    EnsureTyped,
    ToTensord,
    AddChanneld,
    AsChannelFirstd,
)
from monai.data import DataLoader, Dataset, CacheDataset, load_decathlon_datalist, partition_dataset
from monai.utils import first

# -------------------------------------------------------------------------
# 1️⃣ Helper to collect image/label paths per patient
# -------------------------------------------------------------------------
def get_subjects(root: str, split: str = 'train') -> List[Dict[str, str]]:
    '''
    Walk ``root/<split>`` and return a list of dictionaries:

    * training case → {'image': [t1, t1ce, t2, flair], 'label': seg}
    * validation case → {'image': [t1, t1ce, t2, flair]}

    Parameters
    ----------
    root : str
        Path to ``dataset_root``.
    split : {'train', 'val'}
        Which sub‑folder to scan.

    Returns
    -------
    List[Dict[str, str]]
        List of subjects compatible with MONAI’s dictionary‐based datasets.
    '''
    split_dir = os.path.join(root, split)
    subjects = []

    # Each patient lives in its own folder
    patient_dirs = sorted([d for d in glob(os.path.join(split_dir, '*')) if os.path.isdir(d)])

    for pdir in patient_dirs:
        patient_id = os.path.basename(pdir)

        # Build the four modality paths (order matters for channel ordering)
        modalities = ['t1', 't1ce', 't2', 'flair']
        image_paths = [
            os.path.join(pdir, f'{patient_id}_{mod}.nii.gz') for mod in modalities
        ]

        # sanity‑check: all files must exist
        if not all(os.path.isfile(p) for p in image_paths):
            raise FileNotFoundError(f'Missing modality file for {patient_id}')

        subject_dict = {'image': image_paths}

        # Segmentation only exists for training cases
        if split == 'train':
            seg_path = os.path.join(pdir, f'{patient_id}_seg.nii.gz')
            if not os.path.isfile(seg_path):
                raise FileNotFoundError(f'Missing segmentation for {patient_id}')
            subject_dict['label'] = seg_path

        subjects.append(subject_dict)

    return subjects


# -------------------------------------------------------------------------
# 2️⃣ Transforms (train & val)
# -------------------------------------------------------------------------
def get_train_transforms() -> Compose:
    '''
    Typical training transforms:
      * Load NIfTI → torch tensor (channel‑first)
      * Resample to common spacing (optional)
      * Re‑orient to RAS
      * Intensity scaling (0‑1 per channel)
      * RandomCrop / RandomFlip / RandomAffine … (add as needed)
      * EnsureTyped / ToTensord for the final output
    '''
    return Compose(
        [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),  # makes (C, H, W, D)
            Spacingd(
                keys=['image', 'label'],
                pixdim=(1.0, 1.0, 1.0),
                mode=('bilinear', 'nearest'),
            ),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            ScaleIntensityd(keys=['image']),  # 0‑1 per channel
            # ---- add any data‑augmentation here, e.g. RandomFlipd, RandCropByLabeld ----
            EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
            ToTensord(keys=['image', 'label']),
        ]
    )


def get_val_transforms() -> Compose:
    '''
    Validation transforms – deterministic, no augmentation.
    '''
    return Compose(
        [
            LoadImaged(keys=['image', 'label'] if 'label' in keys else ['image']),
            EnsureChannelFirstd(keys=['image', 'label'] if 'label' in keys else ['image']),
            Spacingd(
                keys=['image', 'label'] if 'label' in keys else ['image'],
                pixdim=(1.0, 1.0, 1.0),
                mode=('bilinear', 'nearest'),
            ),
            Orientationd(keys=['image', 'label'] if 'label' in keys else ['image'], axcodes='RAS'),
            ScaleIntensityd(keys=['image']),
            EnsureTyped(keys=['image', 'label'] if 'label' in keys else ['image'], dtype=torch.float32),
            ToTensord(keys=['image', 'label'] if 'label' in keys else ['image']),
        ]
    )


# -------------------------------------------------------------------------
# 3️⃣ Build Datasets & DataLoaders
# -------------------------------------------------------------------------
def build_dataloaders(
    root: str,
    batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    '''
    Returns (train_loader, val_loader).
    '''
    # ---------- subjects ----------
    train_subjects = get_subjects(root, split='train')
    val_subjects = get_subjects(root, split='val')

    # ---------- transforms ----------
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    # ---------- Datasets ----------
    # CacheDataset is fast for medium‑size data; switch to plain Dataset if RAM is limited
    train_dataset = CacheDataset(
        data=train_subjects,
        transform=train_transforms,
        cache_rate=1.0,          # cache whole training set (adjust if memory‑constrained)
        num_workers=num_workers,
    )
    val_dataset = CacheDataset(
        data=val_subjects,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=num_workers,
    )

    # ---------- DataLoaders ----------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


# -------------------------------------------------------------------------
# 4️⃣ Example usage (you can delete / comment this block in production)
# -------------------------------------------------------------------------
if __name__ == '__main__':
    DATA_ROOT = '/path/to/dataset_root'

    train_loader, val_loader = build_dataloaders(
        root=DATA_ROOT,
        batch_size=2,
        num_workers=4,
    )

    # Peek at one batch
    batch = first(train_loader)
    images, labels = batch['image'], batch['label']
    print(f'Image batch shape : {images.shape}')   # (B, C, H, W, D)
    print(f'Label batch shape : {labels.shape}')   # (B, 1, H, W, D)
```

### How it works

| Step | What it does |
|------|--------------|
| **`get_subjects`** | Scans `train/` or `val/`, builds a dict per patient where `'image'` is a **list of four modality files** (ordered as `t1, t1ce, t2, flair`). For training it also adds `'label'` pointing to the segmentation mask. |
| **Transforms** | ‑ `LoadImaged` reads the NIfTI files into NumPy arrays.<br>‑ `EnsureChannelFirstd` guarantees that each modality becomes a separate channel (`C, H, W, D`).<br>‑ `Spacingd` & `Orientationd` bring every case to a common voxel spacing and orientation (important for deep‑learning consistency).<br>‑ `ScaleIntensityd` normalises each channel to `[0, 1]`.<br>‑ (optional) you can insert data‑augmentation ops such as `RandFlipd`, `RandCropByLabeld`, etc. |
| **`CacheDataset`** | Loads all files into RAM once (fast for a few‑hundred patients). Change `cache_rate` or switch to `Dataset` if you hit memory limits. |
| **`DataLoader`** | Provides PyTorch‑compatible batches (`torch.Tensor`) with shape `(B, C, H, W, D)`. This matches the input format expected by `SegResNet`. |

### Plug‑in to a `SegResNet` training loop

```python
from monai.networks.nets import SegResNet

model = SegResNet(
    spatial_dims=3,
    init_filters=32,
    in_channels=4,        # four MRI modalities
    out_channels=2,       # e.g. background + tumor
    dropout_prob=0.2,
    act='PRELU',
)

# typical PyTorch training skeleton …
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn   = torch.nn.CrossEntropyLoss()          # or DiceLoss, etc.

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        imgs = batch['image']   # (B,4,H,W,D)
        segs = batch['label']   # (B,1,H,W,D)
        preds = model(imgs)
        loss = loss_fn(preds, segs.squeeze(1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # validation loop …
```

Feel free to adapt the script:

* **Add more augmentations** (blur, motion, intensity jitter) for robustness.  
* **Change `cache_rate`** if you have many GB of data.  
* **Alter spacing/orientation** to match the specifics of your acquisition protocol.  

That’s everything needed to spin up a MONAI dataloader ready for training a `SegResNet` on the described NIfTI dataset. Happy training!