**Model Genesis & Brain‑Tumor MRI Segmentation – A Practical Workflow with Local NIfTI Volumes**

Model Genesis is a self‑supervised learning (SSL) framework that pre‑trains a generic 3‑D convolutional encoder on a large, unlabeled collection of medical volumes. Once the encoder has learned rich, domain‑specific representations, it can be fine‑tuned for downstream tasks such as brain‑tumor segmentation. Below is a step‑by‑step guide on how to apply Model Genesis to locally‑collected NIfTI MRI scans.

---

## 1. Overview of the Two‑Stage Strategy  

| Stage | Goal | Typical Architecture | Key Operations |
|------|------|----------------------|----------------|
| **Pre‑training (Model Genesis)** | Learn generic 3‑D features from *unlabeled* MRIs. | 3‑D encoder‑decoder (e.g., ResNet‑50‑3D backbone + up‑sampling decoder). | Self‑supervised tasks: **restoration** (in‑painting, non‑linear intensity transformation, local shuffling, etc.). |
| **Fine‑tuning (Segmentation)** | Transfer the pre‑learned encoder to a **segmentation** network. | U‑Net, Attention‑U‑Net, or DeepMedic with the Genesis encoder as the “contracting” path. | Supervised training on **labeled** tumor masks (e.g., BraTS‑style WT/TC/ET). |

The pre‑training stage can be done once on any large pool of brain MRIs (even if they come from different scanners). After that, you only need a relatively small labeled set of tumor cases to achieve high performance.

---

## 2. Preparing Local NIfTI Volumes  

| Step | Action | Reason |
|------|--------|--------|
| **2.1. Gather data** | Collect all raw NIfTI files (`.nii`/`.nii.gz`). Include T1, T1c, T2, and FLAIR if available. | Multi‑modal input improves tumour delineation. |
| **2.2. Convert to a common orientation** | Use `nibabel` or `SimpleITK` to re‑orient to RAS (right‑anterior‑superior). | Guarantees consistent voxel ordering for the network. |
| **2.3. Resample to isotropic spacing** | Typical spacing: 1 mm³ (or 0.8 mm³ for high‑resolution data). | 3‑D convolutions assume uniform voxel size; avoids scale bias. |
| **2.4. Intensity normalisation** | - Clip intensities to the 0.5–99.5 percentile per modality.<br>- Z‑score normalise per volume (mean = 0, std = 1). | Removes scanner‑specific bias and stabilises training. |
| **2.5. Brain masking (optional)** | Apply a fast skull‑stripper (e.g., HD‑BET) and mask out non‑brain voxels. | Reduces background noise for the SSL task. |
| **2.6. Patch extraction** | Extract 3‑D patches (e.g., 96³ voxels) with an overlap of 50 % for both pre‑training & fine‑tuning. | Keeps memory requirements within GPU limits while preserving context. |

*Tip:* Store the pre‑processed patches as compressed NumPy (`.npz`) or HDF5 files for fast I/O during training.

---

## 3. Model Genesis Pre‑training  

### 3.1. Self‑Supervised Tasks (used in the original paper)

| Task | Description | Why it helps |
|------|-------------|--------------|
| **In‑painting** | Random cuboids are masked; network reconstructs missing voxels. | Forces learning of long‑range spatial context. |
| **Non‑linear intensity transformation** | Apply a random gamma or histogram warp; network restores original intensity. | Encourages robustness to scanner intensity variations. |
| **Local shuffling** | Randomly permute small blocks inside a patch; network re‑orders them. | Teaches the model to recognise anatomical structures. |
| **Motion blur / Gaussian blur** | Blur a region; network de‑blurs it. | Improves ability to recover fine details. |

A typical pre‑training schedule:

```yaml
batch_size: 4          # 4 patches per GPU (adjust to memory)
learning_rate: 1e-4
optimiser: AdamW
epochs: 200
loss: L1 + SSIM (weighted)
augmentations: [random_flip, rotate_90, elastic_deform]
```

The encoder component (e.g., 3‑D ResNet‑50) is saved after pre‑training. This checkpoint becomes the “Genesis backbone”.

### 3.2. Practical Tips

* **Dataset size:** Even 50–100 unlabeled volumes yield noticeable gains; more data → diminishing returns.
* **Training hardware:** 1–2 GPUs (≥12 GB VRAM) are sufficient; use mixed‑precision (AMP) to speed up.
* **Checkpointing:** Save the encoder weights every 10 epochs; monitor reconstruction loss on a held‑out set.

---

## 4. Fine‑tuning for Tumour Segmentation  

### 4.1. Network Assembly  

```python
# Pseudo‑code (PyTorch Lightning)
class GenesisSegModel(pl.LightningModule):
    def __init__(self, encoder_weights_path):
        super().__init__()
        # Load pretrained encoder
        self.encoder = ResNet3D(pretrained=False)
        self.encoder.load_state_dict(torch.load(encoder_weights_path))
        # Attach decoder (U‑Net style)
        self.decoder = UNetDecoder(num_classes=3)   # WT, TC, ET
        self.loss_fn = DiceCE(loss_weight=0.5)      # Dice + Cross‑Entropy

    def forward(self, x):
        feats = self.encoder(x)
        return self.decoder(feats)
```

*The encoder is **frozen** for the first few epochs (e.g., 10) then unfrozen for end‑to‑end fine‑tuning.*

### 4.2. Training Settings  

| Parameter | Typical Value |
|-----------|---------------|
| **Batch size** | 2–4 patches (GPU memory dependent) |
| **Learning rate** | 1e‑4 (frozen) → 5e‑5 (unfrozen) |
| **Optimizer** | AdamW (weight decay = 1e‑4) |
| **Loss** | Dice + Cross‑entropy (weighted for class imbalance) |
| **Augmentations** | Random rotation (±15°), elastic deformation, intensity scaling, Gaussian noise |
| **Epochs** | 150–200 (early‑stop on validation Dice) |
| **Metrics** | Dice (WT, TC, ET), Hausdorff 95, Sensitivity/Specificity |

### 4.3. Post‑processing  

1. **Soft‑max → argmax** to obtain discrete labels.  
2. **Connected‑component filtering**: remove isolated voxels (< 10 mm³).  
3. **Morphological closing** (radius = 2 voxels) to smooth borders.  

These steps boost the final Dice scores without additional training.

---

## 5. Evaluation on Local Data  

| Metric | Expected Range (after Genesis) |
|--------|---------------------------------|
| **Whole‑tumour Dice** | 0.88 – 0.93 |
| **Tumour‑core Dice** | 0.80 – 0.86 |
| **Enhancing‑tumour Dice** | 0.78 – 0.84 |
| **Hausdorff 95 (mm)** | 4 – 7 |

*Numbers are based on published Genesis‑based experiments on BraTS‑2020/2021; local data may vary slightly.*

Perform **k‑fold cross‑validation** (k = 5) if the labeled set is small (< 30 cases) to obtain robust estimates.

---

## 6. Implementation Resources  

| Resource | Link / Description |
|----------|--------------------|
| **Official Model Genesis repo** (PyTorch) | `https://github.com/MrGiovanni/ModelGenesis` – includes training scripts for the SSL stage. |
| **Brain‑Tumor Segmentation Code** | `https://github.com/wyzy1234/BrainTumorSegmentation` – a U‑Net that can directly accept a Genesis encoder. |
| **NIfTI I/O utilities** | `nibabel`, `SimpleITK` – Python packages for loading/saving NIfTI. |
| **Data handling** | `torchio` – provides 3‑D transforms, patch samplers, and easy multi‑modal stitching. |
| **Visualization** | `ITK‑Snap`, `3D Slicer`, or `matplotlib` (slice‑by‑slice). |

---

## 7. quick “starter” script (Python)

```python
import torch, torch.nn as nn, torch.optim as optim
import torchio as tio
from model_genesis import ResNet3D  # assume repo installed
from unet3d import UNetDecoder        # your decoder implementation

# -------------------------------------------------
# 1. Load pretrained Genesis encoder
# -------------------------------------------------
enc = ResNet3D()
enc.load_state_dict(torch.load('genesis_encoder.pth'))
enc.eval()                # freeze initially

# -------------------------------------------------
# 2. Build segmentation model
# -------------------------------------------------
class GenesisSeg(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = UNetDecoder(num_classes=3)

    def forward(self, x):
        feats = self.encoder(x)
        return self.decoder(feats)

model = GenesisSeg(enc)

# -------------------------------------------------
# 3. Data pipeline (example with TorchIO)
# -------------------------------------------------
train_transform = tio.Compose([
    tio.RescaleIntensity((0, 1)),
    tio.RandomAffine(scales=(0.9, 1.1), degrees=15),
    tio.RandomElasticDeformation(),
    tio.RandomNoise(std=0.01),
    tio.CropOrPad((96, 96, 96)),
])

train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=4)

# -------------------------------------------------
# 4. Training loop (first 10 epochs frozen)
# -------------------------------------------------
criterion = DiceCrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(150):
    if epoch == 10:
        # unfreeze encoder
        for p in model.encoder.parameters():
            p.requires_grad = True
        optimizer.param_groups[0]['lr'] = 5e-5

    model.train()
    for batch in train_loader:
        imgs = batch['image'][tio.DATA].to(device)          # shape: BxCxDxHxW
        masks = batch['label'][tio.DATA].long().to(device) # shape: BxDxHxW

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
```

This skeleton demonstrates the essential steps: load the Genesis encoder, attach a segmentation decoder, apply 3‑D augmentations, and gradually fine‑tune.

---

## 8. Common Pitfalls & How to Avoid Them  

| Issue | Symptom | Remedy |
|-------|----------|--------|
| **Mismatched voxel spacing** | Poor Dice, checker‑board artifacts. | Resample every volume to the *same* isotropic spacing **before** pre‑training. |
| **Insufficient intensity normalisation** | Network diverges early. | Clip extreme percentiles and apply per‑volume Z‑score. |
| **Over‑fitting on small labelled set** | Validation Dice drops after a few epochs. | Use strong augmentation, early‑stop, and/or freeze the encoder longer. |
| **Label leakage between modalities** | Unrealistically high Dice on training set. | Ensure that each patch’s label comes from the *same* spatial region as its image modalities only (no cross‑patch mixing). |
| **GPU out‑of‑memory** | Crash during training. | Reduce patch size (e.g., 64³), enable gradient checkpointing, or use mixed‑precision (`torch.cuda.amp`). |

---

## 9. Summary  

1. **Pre‑train** a 3‑D encoder on all locally‑available NIfTI brain MRIs using Model Genesis’ self‑supervised restoration tasks.  
2. **Prepare** the data (orientation, isotropic resampling, intensity normalisation, optional brain masking) and extract overlapping 3‑D patches.  
3. **Fine‑tune** a segmentation network (U‑Net‑style) that uses the Genesis encoder as its backbone; start with the encoder frozen, then unfreeze for end‑to‑end training.  
4. **Post‑process** the probability maps (connected‑component filtering, morphological ops) to obtain clean tumour masks.  
5. **Validate** with Dice / Hausdorff metrics; expect a 3‑5 % Dice boost compared to training from scratch, especially when labeled data are scarce.  

By leveraging Model Genesis, you obtain a strong, data‑efficient starting point that adapts quickly to the specific characteristics of your local NIfTI brain‑MRI collection, leading to robust tumor segmentation with relatively modest annotation effort.