# eval_stage2_vit.py
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import classification_report, confusion_matrix

from src.config import *

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# Transforms (validation only)
# --------------------------------------------------
val_transform = transforms.Compose([
    transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Dataset & Loader
# --------------------------------------------------
val_ds = datasets.ImageFolder(STAGE2_VAL_DIR, transform=val_transform)

val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

class_names = val_ds.classes
print("Stage-2 classes:", class_names)

# --------------------------------------------------
# Load ViT Model
# --------------------------------------------------
model = create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=len(class_names)
)

model.load_state_dict(
    torch.load(STAGE2_VIT_MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --------------------------------------------------
# Metrics
# --------------------------------------------------
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = (all_preds == all_labels).mean()
print(f"\n✅ ViT Stage-2 Accuracy: {accuracy * 100:.2f}%\n")

print("📊 Classification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )
)

print("🧩 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
