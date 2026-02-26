Here's a **refactored and integrated version** of your code that improves:

- **Reusability and consistency** between train/val transforms and data loading
- **Robustness**: avoids duplication, uses `Optional`, and handles missing labels gracefully
- **Extensibility**: easier to add new modalities or transforms
- **Clarity**: separates configuration, transforms, and data loading logic

### ✅ Key Improvements:
- ✅ Unified transform construction with `mode` and `dtype` per key
- ✅ Reusable helper `get_transforms()` for train/val distinction
- ✅ `get_subjects()` now returns **label-only when available** (i.e., no `'label'` key for val)
- ✅ Uses `Config`-style constants for better maintainability
- ✅ Uses `type hints`, `Literal`, `Optional`, `Path` for safety
- ✅ Avoids duplicate code via `@dataclass` for subject dicts (optional but clean)

---

### ✅ Refactored Code

```python
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass

import os
import glob
import torch
from monai.data import Dataset, CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityd,
    EnsureTyped,
    ToTensord,
)

# Constants
TRAIN_DIR = 'train'
VAL_DIR = 'val'
MODALITIES = ('t1', 't1ce', 't2', 'flair')  # ordered for channel consistency

# Optional: use dataclass for subject dict (not strictly needed, but nice for type clarity)
@dataclass
class SubjectDict:
    image: List[str]
    label: Optional[str] = None


def get_subjects(root: str, split: Literal['train', 'val']) -> List[SubjectDict]:
    '''
    Walk `root/<split>` and return a list of subject dicts.
    - Training: includes `'label'` key
    - Validation: omits `'label'` key entirely
    '''
    split_path = Path(root) / split
    subjects: List[SubjectDict] = []

    patient_dirs = sorted([d for d in split_path.glob('*') if d.is_dir()])

    for pdir in patient_dirs:
        patient_id = pdir.name
        image_paths = [str(pdir / f'{patient_id}_{mod}.nii.gz') for mod in MODALITIES]

        # Ensure all modalities exist
        missing = [p for p in image_paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f'Missing modality files for '{patient_id}': {missing}'
            )

        subject = SubjectDict(image=image_paths)

        if split == 'train':
            seg_path = pdir / f'{patient_id}_seg.nii.gz'
            if not seg_path.exists():
                raise FileNotFoundError(f'Missing segmentation for '{patient_id}'')
            subject.label = str(seg_path)

        subjects.append(subject)

    return subjects


def get_transforms(
    stage: Literal['train', 'val'],
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    cache: bool = False,  # only used for future extensibility (e.g., dynamic transforms)
) -> Compose:
    '''
    Return appropriate transforms for train/val stage.
    - For training: includes augmentation (placeholder)
    - For validation: deterministic only
    '''
    image_keys = ['image']
    label_keys = ['label']

    # Determine which keys to load (label only if stage == train)
    load_keys = image_keys + (label_keys if stage == 'train' else [])

    base_transforms = [
        LoadImaged(keys=load_keys),
        EnsureChannelFirstd(keys=load_keys),
        Spacingd(
            keys=load_keys,
            pixdim=pixdim,
            mode=('bilinear', 'nearest') if stage == 'train' else ('bilinear', 'nearest'),
        ),
        Orientationd(keys=load_keys, axcodes='RAS'),
        ScaleIntensityd(keys=image_keys),
        EnsureTyped(
            keys=load_keys,
            dtype={
                'image': torch.float32,
                'label': torch.long,
            } if 'label' in load_keys else {'image': torch.float32},
        ),
        ToTensord(keys=load_keys),
    ]

    # Train-only augmentations (expand as needed)
    if stage == 'train':
        # Example augmentations — adapt per your use case
        from monai.transforms import (
            RandFlipd,
            RandRotate90d,
            RandAffined,
            RandCropByPosNegLabeld,
        )
        augmentations = [
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=['image', 'label'], prob=0.5),
            RandAffined(
                keys=['image', 'label'],
                prob=0.5,
                translate_range=(10, 10, 10),
                rotate_range=(0.1, 0.1, 0.1),
            ),
            # Assuming label-based cropping; adapt if needed
            RandCropByPosNegLabeld(
                keys=['image', 'label'],
                label_key='label',
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=2,
            ),
        ]
        transforms = base_transforms + augmentations
    else:
        transforms = base_transforms

    return Compose(transforms)


def build_dataloaders(
    root: str,
    batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_rate: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    '''
    Build train and val dataloaders.
    '''
    train_subjects = get_subjects(root, split=TRAIN_DIR)
    val_subjects = get_subjects(root, split=VAL_DIR)

    train_transforms = get_transforms(stage='train')
    val_transforms = get_transforms(stage='val')

    # Use CacheDataset by default; fallback to Dataset if cache_rate=0.0
    train_dataset = CacheDataset(
        data=train_subjects,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    val_dataset = Dataset(  # Use plain Dataset for val to save memory (or CacheDataset with lower cache_rate)
        data=val_subjects,
        transform=val_transforms,
    )

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
```

---

### 🚀 Usage Example

```python
from pathlib import Path

root = Path('/path/to/brats/dataset')  # expected layout: root/train/*, root/val/*
train_loader, val_loader = build_dataloaders(str(root), batch_size=4)

for batch in train_loader:
    images = batch['image']  # shape: [B, C=4, H, W, D]
    labels = batch['label']  # shape: [B, H, W, D], int64
    # ... train model
```

---

### 🔁 Future-Proofing Tips

- To add **test** data: extend `get_subjects()` to return only images (no labels) and update `get_transforms()` accordingly.
- To support **more modalities**: just change `MODALITIES` tuple.
- To add **new augmentations**: insert in `train` block in `get_transforms()`.
- To reduce memory usage: set `cache_rate=0.5` or use `Dataset` instead of `CacheDataset`.

Let me know if you’d like to:
- add mixed-precision support (`torch.cuda.amp`)
- integrate label postprocessing (e.g., `ConvertToMultiChannelBasedOnBratsClassesd`)
- output sample-wise metadata (e.g., `patient_id`)  
I'm happy to extend further!

# Version 2 Bug Fixes
You're encountering this error:

```
ValueError: Sequence must have length 1, got 2.
```

**Most likely cause**: In your `get_transforms(stage='val')`, you're passing **a sequence of `mode` tuples with length 2** to `Spacingd`, but `Spacingd` expects **one mode per key**, and since `load_keys = ['image']` for validation, you're passing `mode=('bilinear', 'nearest')` — i.e., two modes for one key.

---

### 🔍 Let's fix the `Spacingd` call

In `get_transforms`, this line is problematic for val:

```python
Spacingd(
    keys=load_keys,
    pixdim=pixdim,
    mode=('bilinear', 'nearest'),  # ❌ Problem: length 2, but load_keys has length 1 for val
)
```

- `keys=['image']` → needs `mode=('bilinear',)`  
- `keys=['image', 'label']` → needs `mode=('bilinear', 'nearest')`

✅ **Solution**: Dynamically build `mode` as a tuple of the right length.

Also: for `Orientationd`, same issue — `axcodes` is fine (single string), but other params like `mode` (if used) would need care.

Here's the **minimal, correct fix** to `get_transforms`:

---

### ✅ Fixed `get_transforms` (only the changed section)

Replace your `Spacingd` and `Orientationd` calls inside `get_transforms` with:

```python
# Build mode list: 'bilinear' for images, 'nearest' for labels
num_keys = len(load_keys)
image_keys_set = {'image'}
mode_list = [
    'bilinear' if k in image_keys_set else 'nearest'
    for k in load_keys
]

# Similarly for Orientationd (if you ever specify mode there — but default is fine)
# Orientationd defaults to 'bilinear' for all keys, so no need to override unless necessary.

base_transforms = [
    LoadImaged(keys=load_keys),
    EnsureChannelFirstd(keys=load_keys),
    Spacingd(
        keys=load_keys,
        pixdim=pixdim,
        mode=mode_list,  # ✅ Correct length now
    ),
    Orientationd(keys=load_keys, axcodes='RAS'),  # ✅ axcodes is always fine as str
    ScaleIntensityd(keys=['image']),
    EnsureTyped(
        keys=load_keys,
        dtype={
            'image': torch.float32,
            'label': torch.long,
        } if 'label' in load_keys else {'image': torch.float32},
    ),
    ToTensord(keys=load_keys),
]
```

> 🔎 Why this works:
> - For `stage='val'`: `load_keys = ['image']` → `mode_list = ['bilinear']`
> - For `stage='train'`: `load_keys = ['image', 'label']` → `mode_list = ['bilinear', 'nearest']`

---

### 🛠 Full corrected `get_transforms`

```python
def get_transforms(
    stage: Literal['train', 'val'],
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    image_keys = ['image']
    label_keys = ['label']

    # Determine which keys to load
    load_keys = image_keys + (label_keys if stage == 'train' else [])
    
    # 🔑 Fixed: Dynamic mode list based on key type
    image_keys_set = {'image'}
    mode_list = [
        'bilinear' if k in image_keys_set else 'nearest'
        for k in load_keys
    ]

    base_transforms = [
        LoadImaged(keys=load_keys),
        EnsureChannelFirstd(keys=load_keys),
        Spacingd(
            keys=load_keys,
            pixdim=pixdim,
            mode=mode_list,
        ),
        Orientationd(keys=load_keys, axcodes='RAS'),
        ScaleIntensityd(keys=image_keys),  # Only scale image, not label
        EnsureTyped(
            keys=load_keys,
            dtype={
                'image': torch.float32,
                'label': torch.long,
            } if 'label' in load_keys else {'image': torch.float32},
        ),
        ToTensord(keys=load_keys),
    ]

    if stage == 'train':
        from monai.transforms import (
            RandFlipd,
            RandRotate90d,
            RandAffined,
            RandCropByPosNegLabeld,
        )
        augmentations = [
            RandFlipd(keys=load_keys, prob=0.5, spatial_axis=0),
            RandRotate90d(keys=load_keys, prob=0.5),
            RandAffined(
                keys=load_keys,
                prob=0.5,
                translate_range=(10, 10, 10),
                rotate_range=(0.1, 0.1, 0.1),
            ),
            RandCropByPosNegLabeld(
                keys=load_keys,
                label_key='label',
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=2,
            ),
        ]
        transforms = base_transforms + augmentations
    else:
        transforms = base_transforms

    return Compose(transforms)
```

---

### ✅ Bonus: Double-check `get_subjects`

Ensure `get_subjects(..., split=VAL_DIR)` returns only `'image'` filenames (no label paths), e.g.:

```python
def get_subjects(root: str, split: str) -> List[Dict[str, str]]:
    split_path = Path(root) / split
    subjects = []
    for case_dir in split_path.glob('*/'):
        if case_dir.is_dir():
            subjects.append({
                'image': str(case_dir / f'{case_dir.name}_t1ce.nii.gz'),  # or whatever modality(s)
                # no 'label' for val!
            })
    return subjects
```

If your val set *does* have labels (e.g., for validation + metric computation), then use `load_keys = ['image', 'label']`, and `mode_list` will correctly grow.

---

Let me know if you'd like to:
- Make `mode` selection more robust (e.g., per-key override)
- Support multi-modality images (e.g., T1, T2, FLAIR, T1ce)
- Add label post-processing for BraTS classes (`BratsLabelsd`)

Happy to help refine further! 🧠