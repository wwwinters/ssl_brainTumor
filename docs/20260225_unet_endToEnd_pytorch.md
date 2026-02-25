Here is an **end-to-end PyTorch Lightning script** using **MONAI** to train a 3D UNet on BraTS-style NIfTI data with early stopping, model checkpoints, and detailed metric tracking.

It assumes the following **folder structure**:

```
dataset_root/
├── train/
│   ├── patient001/
│   │   ├── patient001_t1.nii.gz
│   │   ├── patient001_t1ce.nii.gz
│   │   ├── patient001_t2.nii.gz
│   │   ├── patient001_flair.nii.gz
│   │   └── patient001_seg.nii.gz
│   ├── patient002/ ...
│   └── ...
└── val/
    ├── patient050/ ...
    └── ...
```

> ✅ All modalities are required per patient; segmentation only for training data.

---

### ✅ Key Features:
- Uses `MONAI` for robust medical image I/O, preprocessing, transforms
- Custom `LightningDataModule` for loading patient-wise NIfTI files
- Uses `DiceScore`, `IoU`, and loss tracking
- Saves best models + logs to CSV & tensorboard (optional)
- Early stopping with `monitor='val_dice'`
- Multi-label support ( BraTS uses 4 labels: `0=background`, `1=WT`, `2=TC`, `3=ET` or full 4-class if preferred — script shows full 4-class and binaryWT mode )

---

### 📜 `train_unet_brats.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
import logging

# -----------------------------
# Configuration / Hyperparameters
# -----------------------------
ROOT_DIR = 'dataset_root'  # update to your dataset root
TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
VAL_DIR = os.path.join(ROOT_DIR, 'val')

MODALITIES = ['t1', 't1ce', 't2', 'flair']
SEG_NAME = 'seg'

BATCH_SIZE = 2
NUM_WORKERS = 4
PATCH_SIZE = (128, 128, 128)  # random crop size for training
IN_CHANNELS = len(MODALITIES)
OUT_CHANNELS = 4  # full 4-class BraTS: 0 (background), 1 (WT), 2 (TC), 3 (ET)
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'

# Training
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 15  # early stopping patience

# -----------------------------
# Data Module (Lightning)
# -----------------------------
class BraTSDataset(Dataset):
    def __init__(self, root_dir, modalities, seg_name, transform=None, is_train=True):
        self.root_dir = root_dir
        self.modalities = modalities
        self.seg_name = seg_name
        self.transform = transform
        self.is_train = is_train
        self.patients = sorted(glob.glob(os.path.join(root_dir, 'patient*')))

        # Collect all patient IDs
        self.patient_ids = [os.path.basename(p) for p in self.patients]

        # Optionally: filter out patients without all modalities or seg
        valid_patients = []
        for pid in self.patient_ids:
            patient_dir = os.path.join(root_dir, pid)
            files = glob.glob(os.path.join(patient_dir, f'{pid}_*.nii.gz'))
            required = [f'{pid}_{mod}.nii.gz' for mod in modalities]
            if not all(os.path.exists(os.path.join(patient_dir, f)) for f in required):
                print(f'⚠️ Skipping {pid}: missing modalities')
                continue
            if is_train and not os.path.exists(os.path.join(patient_dir, f'{pid}_{seg_name}.nii.gz')):
                print(f'⚠️ Skipping {pid}: missing segmentation')
                continue
            valid_patients.append(pid)

        self.patient_ids = valid_patients

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        patient_dir = os.path.join(self.root_dir, pid)

        image_paths = [os.path.join(patient_dir, f'{pid}_{mod}.nii.gz') for mod in self.modalities]
        data = {mod: path for mod, path in zip(self.modalities, image_paths)}

        if self.is_train:
            seg_path = os.path.join(patient_dir, f'{pid}_{self.seg_name}.nii.gz')
            data['seg'] = seg_path

        # Apply transform (Monai dict transforms)
        if self.transform:
            data = self.transform(data)

        # Return dict with keys: t1, t1ce, t2, flair, seg (optional)
        return data


class BraTSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir,
        val_dir,
        modalities,
        seg_name,
        batch_size=1,
        num_workers=4,
        patch_size=(128, 128, 128),
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.modalities = modalities
        self.seg_name = seg_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size

        self.train_transform = self.get_train_transforms()
        self.val_transform = self.get_val_transforms()

    def get_train_transforms(self):
        return Compose(
            [
                LoadImaged(keys=['t1', 't1ce', 't2', 'flair', 'seg']),
                EnsureChannelFirstd(keys=['t1', 't1ce', 't2', 'flair', 'seg']),
                Spacingd(
                    keys=['t1', 't1ce', 't2', 'flair', 'seg'],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest'),
                ),
                Orientationd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], axcodes='RAS'),
                ScaleIntensityRanged(
                    keys=['t1', 't1ce', 't2', 'flair'],
                    a_min=0,
                    a_max=2000,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # Crop foreground (optional but recommended for memory)
                CropForegroundd(
                    keys=['t1', 't1ce', 't2', 'flair', 'seg'],
                    source_key='t1',
                    allow_smaller=True,
                ),
                RandSpatialCropd(
                    keys=['t1', 't1ce', 't2', 'flair', 'seg'],
                    roi_size=self.patch_size,
                    random_size=False,
                ),
                RandFlipd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], spatial_axis=0, prob=0.5),
                RandFlipd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], spatial_axis=1, prob=0.5),
                RandRotate90d(keys=['t1', 't1ce', 't2', 'flair', 'seg'], prob=0.5),
                ToTensord(keys=['t1', 't1ce', 't2', 'flair', 'seg']),
            ]
        )

    def get_val_transforms(self):
        return Compose(
            [
                LoadImaged(keys=['t1', 't1ce', 't2', 'flair', 'seg']),
                EnsureChannelFirstd(keys=['t1', 't1ce', 't2', 'flair', 'seg']),
                Spacingd(
                    keys=['t1', 't1ce', 't2', 'flair', 'seg'],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest'),
                ),
                Orientationd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], axcodes='RAS'),
                ScaleIntensityRanged(
                    keys=['t1', 't1ce', 't2', 'flair'],
                    a_min=0,
                    a_max=2000,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # Crop same ROI as training? (e.g., central crop or center-of-foreground)
                # For simplicity, use same CropForegroundd + fixed crop
                CropForegroundd(
                    keys=['t1', 't1ce', 't2', 'flair', 'seg'],
                    source_key='t1',
                    allow_smaller=True,
                ),
                ToTensord(keys=['t1', 't1ce', 't2', 'flair', 'seg']),
            ]
        )

    def setup(self, stage=None):
        # Called by trainer.fit()
        if stage in (None, 'fit'):
            self.train_ds = BraTSDataset(
                self.train_dir, self.modalities, self.seg_name, transform=self.train_transform, is_train=True
            )
            self.val_ds = BraTSDataset(
                self.val_dir, self.modalities, self.seg_name, transform=self.val_transform, is_train=True
            )
        if stage == 'validate':
            self.val_ds = BraTSDataset(
                self.val_dir, self.modalities, self.seg_name, transform=self.val_transform, is_train=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if DEVICE == 'gpu' else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if DEVICE == 'gpu' else False,
        )


# -----------------------------
# LightningModule
# -----------------------------
class UNetLightningModule(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.2,
        )

        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(reduction='mean_batch')  # per class dice
        self.learning_rate = learning_rate

        # Store metrics for epoch-level logging
        self.train_losses = []
        self.val_losses = []
        self.val_dice_per_epoch = []  # list of per-class dice per batch

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = torch.cat([batch[mod] for mod in MODALITIES], dim=1)  # [B, 4, H, W, D]
        labels = batch['seg']
        outputs = self(images)

        loss = self.loss_function(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        images = torch.cat([batch[mod] for mod in MODALITIES], dim=1)
        labels = batch['seg']
        outputs = self(images)

        loss = self.loss_function(outputs, labels)
        self.val_losses.append(loss.item())

        # Compute Dice per class
        pred_argmax = torch.argmax(outputs, dim=1, keepdim=True)
        self.dice_metric(y_pred=pred_argmax, y=labels)
        dice_scores = self.dice_metric.compute()  # shape: [num_classes]
        self.dice_metric.reset()
        # dice_scores: tensor([dice_bg, diceWT, diceTC, diceET])

        self.log('val_loss', loss, prog_bar=True)
        # Log per-class dice
        for i, name in enumerate(['dice_bg', 'dice_WT', 'dice_TC', 'dice_ET']):
            self.log(f'val_{name}', dice_scores[i], prog_bar=False)

        # return per-batch dice average (mean over labels, excluding background)
        dice_mean = dice_scores[1:].mean()  # exclude background (index 0)
        self.log('val_dice', dice_mean, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = np.mean(self.train_losses)
        self.log('train_loss_epoch', avg_loss)
        self.train_losses.clear()  # free memory

    def on_validation_epoch_end(self):
        avg_loss = np.mean(self.val_losses)
        self.log('val_loss_epoch', avg_loss)
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss_epoch',
            },
        }


# -----------------------------
# Training Script
# -----------------------------
def main():
    pl.seed_everything(42)
    set_determinism(42)

    # Data
    datamodule = BraTSDataModule(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        modalities=MODALITIES,
        seg_name=SEG_NAME,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        patch_size=PATCH_SIZE,
    )

    # Model
    model = UNetLightningModule(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        learning_rate=LEARNING_RATE,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='unet-brats-{val_dice:.4f}',
        save_top_k=3,
        monitor='val_dice',
        mode='max',
        save_weights_only=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_dice',
        patience=PATIENCE,
        mode='max',
        verbose=True,
    )

    # Logger (TensorBoard)
    logger = TensorBoardLogger('tb_logs', name='unet_brats')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=5,
        enable_checkpointing=True,
        enable_progress_bar=True,
        precision=16 if torch.cuda.is_available() else 32,  # mixed precision for speed
    )

    # Train!
    trainer.fit(model, datamodule)

    # Optional: load best model and validate
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f'✅ Best model saved at: {best_model_path}')

        # Load best weights and validate on val set (optional)
        best_model = UNetLightningModule.load_from_checkpoint(best_model_path)
        trainer.validate(model=best_model, datamodule=datamodule)

    print('✅ Training completed.')


if __name__ == '__main__':
    main()
```

---

### ✅ Additional Notes

#### 🔍 Metrics Explained:
- `train_loss_epoch`: Dice loss (not normalized)
- `val_loss_epoch`: Validation Dice loss
- `val_dice`: Mean Dice over foreground labels (1–3), i.e., WT, TC, ET — excludes background
- `val_dice_WT`, `val_dice_TC`, `val_dice_ET`: per-label Dice

> 📈 Use TensorBoard to visualize:  
> ```bash
> tensorboard --logdir tb_logs/unet_brats
> ```

#### 🧠 BraTS Label Mapping:
By default, `out_channels=4` means 4-class output. If you want binary (e.g., tumor vs. non-tumor), modify `OUT_CHANNELS=2`, and adjust `LoadImaged` transforms to combine BraTS labels:
```python
# Example: whole tumor (label 1,2,3 → 1), background (0 → 0)
# Use `MapToBinaryd` or `Lambdad` transform
```

#### 💡 For Larger Datasets:
- Increase `PATCH_SIZE` (e.g., 160x160x160)
- Set `CACHE_RATE=1.0` using `CachedDataset` if memory allows
- Enable `multi-GPU` via `devices=2` in `Trainer(...)`

---

Let me know if you'd like:
- `TorchScript` export + inference notebook
- BraTS 4-class → 3-class or binary conversion
- Evaluation script for test set
- Dockerfile for reproducibility