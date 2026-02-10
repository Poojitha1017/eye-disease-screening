# train_stage2_vit.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from tqdm import tqdm
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

from src.config import *

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# Transforms (STRONGER for ViT)
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Dataset & Loaders
# --------------------------------------------------
train_ds = datasets.ImageFolder(STAGE2_TRAIN_DIR, transform=train_transform)
val_ds   = datasets.ImageFolder(STAGE2_VAL_DIR, transform=val_transform)

print("Stage-2 classes:", train_ds.classes)

train_loader = DataLoader(
    train_ds,
    batch_size=max(1, STAGE2_BATCH_SIZE // 2),
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=max(1, STAGE2_BATCH_SIZE // 2),
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# Model: Vision Transformer (ViT)
# --------------------------------------------------
num_classes = len(train_ds.classes)

model = create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=num_classes
)
model.to(device)

# --------------------------------------------------
# Class-Weighted Loss (IMBALANCE + DR BOOST)
# --------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_ds.targets),
    y=train_ds.targets
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# 🔥 IMPORTANT: Boost DR penalty (index 0 = DR)
class_weights[0] *= 1.5

criterion = nn.CrossEntropyLoss(weight=class_weights)

# --------------------------------------------------
# Optimizer (ViT-specific)
# --------------------------------------------------
vit_lr = globals().get("STAGE2_VIT_LR", 3e-5)
vit_epochs = globals().get("STAGE2_VIT_EPOCHS", STAGE2_EPOCHS)

optimizer = optim.AdamW(
    model.parameters(),
    lr=vit_lr,
    weight_decay=1e-4
)

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
best_val_loss = float("inf")

os.makedirs(os.path.dirname(STAGE2_VIT_MODEL_PATH), exist_ok=True)

for epoch in range(vit_epochs):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(
        train_loader, desc=f"ViT Stage-2 Epoch {epoch+1}/{vit_epochs}"
    ):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # --------------------------------------------------
    # Save Best Model
    # --------------------------------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), STAGE2_VIT_MODEL_PATH)
        print("✔ Saved best ViT Stage-2 model")

print("✅ ViT Stage-2 training complete")




