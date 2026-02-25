Yes! You can **fully train a PyTorch U-Net without MONAI** — MONAI is helpful for medical imaging but not required. Below is a **complete, self-contained training pipeline** using only PyTorch (and optionally PyTorch Lightning), with no external medical-imaging dependencies.

We’ll cover:
1. ✅ U-Net model definition (minimal, production-ready)  
2. ✅ Custom dataset (e.g., loading from folders or synthetic data)  
3. ✅ Loss functions (`DiceLoss`, `BCEWithLogitsLoss`) — **no MONAI needed**  
4. ✅ Training loop (pure PyTorch or with Lightning)  
5. ✅ Data augmentation (using `torchvision.transforms`)

---

## ✅ 1. Clean U-Net (PyTorch only)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        features = [init_features * 2**i for i in range(4)]  # [32, 64, 128, 256]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # up-conv
            skip = skip_connections[i // 2]

            # ⚠️ Crop and concatenate: ensure same spatial size
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)
```

> ✅ **No MONAI. No external medical libraries.** Just pure PyTorch.

---

## ✅ 2. Loss Functions (Custom Implementations)

### 🔹 Dice Loss (for binary segmentation)

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: logits or sigmoid probs (N, 1, H, W)
        # target: (N, 1, H, W) in {0,1}
        if pred.dim() > target.dim():
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
        union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
```

### 🔹 Combined Loss: BCE + Dice

```python
bce = nn.BCEWithLogitsLoss()
dice = DiceLoss()

# In training step:
loss = bce(logits, targets) + dice(logits, targets)
```

> ✅ Works with raw logits (no need to apply sigmoid first in `DiceLoss` if you handle it carefully — above version uses `torch.sigmoid(pred)` inside the loss).

---

## ✅ 3. Dataset (e.g., Images & Masks in Folders)

Assume structure:
```
data/
  train/
    images/
      img001.png
      img002.png
    masks/
      img001.png  # same name, 0/1-valued mask
      img002.png
  val/
    images/
    masks/
```

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.filenames[idx])

        image = Image.open(img_path).convert('RGB')  # or 'L' if grayscale
        mask = Image.open(mask_path).convert('L')    # mask is grayscale

        if self.transform:
            seed = torch.random.seed() if not hasattr(self, '_seed') else self._seed
            torch.random.manual_seed(seed)
            image = self.transform(image)
            torch.random.manual_seed(seed)  # same seed for mask
            mask = self.transform(mask)

            # Ensure mask is in {0,1} or {0,...,C-1}
            # For binary: binarize if needed
            if mask.max() > 1:
                mask = (mask > 0.5).float()  # threshold at 0.5
            else:
                mask = mask.float() / 255.0  # normalize to [0,1]

        return image, mask

# Example transform (resize + tensor + normalize)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])
```

> 🔍 **Note:** You can skip normalization if your data is small or medical (e.g., no ImageNet stats). For grayscale: use `[0.5]`.

---

## ✅ 4. Training Loop (Pure PyTorch)

```python
# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
bce = nn.BCEWithLogitsLoss()
dice = DiceLoss()

# --- Data ---
train_dataset = SegmentationDataset('data/train/images', 'data/train/masks', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# --- Training ---
model.train()
for epoch in range(20):
    epoch_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # logits
        loss = bce(outputs, masks) + dice(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/20, Loss: {epoch_loss/len(train_loader):.4f}')

# Save
torch.save(model.state_dict(), 'unet.pth')
```

✅ Works out of the box — no extra dependencies!

---

## ✅ 5. PyTorch Lightning (Optional, Still MONAI-free)

```python
from lightning.pytorch import LightningModule, Trainer
import torch

class UNetLightning(LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = UNet(in_channels=3, out_channels=1)
        self.lr = lr
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.bce(logits, masks) + self.dice(logits, masks)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Usage:
# dataset, dataloader = ...
# model = UNetLightning()
# trainer = Trainer(max_epochs=20, devices=1)
# trainer.fit(model, train_loader)
```

---

## 🧪 Tips for Better Training

| Tip | Why |
|-----|-----|
| **Use `nn.Upsample` + `conv` instead of transposed convs** | Avoid checkerboard artifacts |
| **Apply `torch.cuda.amp.autocast()`** | Speed + memory (half precision) |
| **Add dropout in decoder (small)** | Reduce overfitting |
| **Use `torchvision.transforms.RandomAffine`, `RandomHorizontalFlip`** | Strong augmentation for small datasets |
| **Monitor IoU/Dice on val set** | Not just loss |

Example augmentation:
```python
augment = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
])
```

> ✅ Remember to use **same random seed** for image & mask transforms to keep alignment.

---

## 🚀 Bonus: Mixed Precision (AMP) for Faster Training

```python
scaler = torch.cuda.amp.GradScaler()

for images, masks in train_loader:
    images, masks = images.to(device), masks.to(device)
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        outputs = model(images)
        loss = bce(outputs, masks) + dice(outputs, masks)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

Let me know if you'd like:
- 🧫 A complete end-to-end example (GitHub-ready repo layout)
- 📊 TensorBoard logging
- 📈 Validation metrics (IoU, Dice per class)
- 🌐 Multi-GPU (DDP) setup (as your original title mentioned!)

Happy to tailor this further! 😊