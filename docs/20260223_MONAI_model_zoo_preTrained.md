MONAI (Medical Open Network for AI) provides a **Model Zoo** that includes pre-trained models, many of which are trained using **self-supervised learning (SSL)** for medical image analysis. As of MONAI v1.3+ (and continuing into v1.4), the following SSL pre-trained models are available from the MONAI Model Zoo (hosted at [https://github.com/Project-MONAI/model-zoo](https://github.com/Project-MONAI/model-zoo)):

---

### ✅ **Available Self-Supervised Pre-trained Models**

| Model | SSL Method | Backbone | Use Case | Download (via `monai.model_zoo`) |
|-------|------------|----------|----------|-----------------------------------|
| **`uni`** | **DINO / ViT** (self-supervised on 10M histopathology images) | ViT-B/16 (224×224), ViT-H/14 (512×512) | General histopathology embedding (tumors, tissue types) | `UNETR`, `UNETR++`, `SwinUNETR` backbones also support `swin_unetr` variants — but `uni` is standalone SSL model |
| **`split_brain`** | **Barlow Twins / BYOL** (on brain MRIs) | ResNet-50 | Pretraining on brain T1w MRI for downstream tasks (e.g., segmentation) | `monai.apps.pathology.models.SplitBrainNet` (via `SplitBrainNet` class) |
| **`chexpert`** | **SimCLR** (on chest X-rays) | ResNet-50 | General chest X-ray representation learning | Not directly available in current MONAI Model Zoo (historically referenced, but not exposed via `model_zoo` API yet) |
| **`spleen_ct_ssl`** | **MAE (Masked Autoencoders)** | ResNet-50 or ViT | Pretraining on abdominal CT (spleen-focused, but generalizable) | ✅ Available via `MONAIModelLoader` with ID `'spleen_ct_ssl'` |
| **`prostate_mri_ssl`** | **Barlow Twins** | ResNet-50 | Pretraining on multiparametric MRI (T2W, DWI, DCE) | ✅ Available under ID `'prostate_mri_ssl'` |

---

### 🔍 How to Load SSL Pretrained Models in MONAI

Use the MONAI Model Zoo API:

```python
from monai.model_zoo import download_model

# Example: Load the spleen CT SSL pre-trained model
model_path = download_model(
    url='models/ssl_spleen_ct_resnet50.zip',  # or use built-in ID
    path='.', 
    progress=True
)

# Alternatively, via model ID (MONAI >= v1.2)
from monai.model_zoo import ModelZoo

# For prostate_mri_ssl:
model = ModelZoo.load_model('prostate_mri_ssl')
```

Or use `MONAIModelLoader`:

```python
from monai.networks.nets import UNETR, SwinUNETR
from monai.model_zoo import download_and_load_checkpoint

# Load SSL backbone checkpoint for fine-tuning
ckpt_path = download_and_load_checkpoint('spleen_ct_ssl')
model = UNETR(
    in_channels=1,
    out_channels=2,
    img_size=(96,96,96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed='perceptron',
    norm_name='instance',
    conv_block=True,
    res_block=True,
    dropout_rate=0.0,
)

model.load_from(ckpt_path)  # Load SSL weights into backbone (e.g., encoder)
```

> 💡 **Note**: Many SSL models (e.g., `split_brain`, `spleen_ct_ssl`) provide **encoder weights** intended to be used as backbones for downstream architectures (e.g., UNETR,SegResNet,ResNet-based classifiers).

---

### 📌 Key Resources

- 📘 MONAI Model Zoo Docs: https://github.com/Project-MONAI/model-zoo  
- 📦 Pretrained SSL models list (with IDs & metrics): https://github.com/Project-MONAI/model-zoo/blob/main/docs/model_cards/ssl/README.md  
- 📚 SSL Methods used: DINO, Barlow Twins, MAE, SimCLR, BYOL  
- 📊 Evaluation performance: See each model card (e.g., `spleen_ct_ssl` yields ~+5% Dice over random init on Spleen CT segmentation)

---

Would you like code examples for fine-tuning one of these (e.g., prostate MRI SSL → prostate segmentation), or help selecting the best SSL model for your use case (e.g., histology vs. MRI vs. CT)?