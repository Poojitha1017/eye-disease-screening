# train_stage1.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

from config import *

# ======================
# Reproducibility
# ======================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================
# Transforms
# ======================
train_transform = transforms.Compose([
    transforms.Resize((STAGE1_IMG_SIZE, STAGE1_IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])

val_transform = transforms.Compose([
    transforms.Resize((STAGE1_IMG_SIZE, STAGE1_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# Datasets
# ======================
train_ds = datasets.ImageFolder(STAGE1_TRAIN_DIR, transform=train_transform)
val_ds   = datasets.ImageFolder(STAGE1_VAL_DIR, transform=val_transform)

print("Class mapping:", train_ds.class_to_idx)

# ======================
# DataLoaders
# ======================
train_loader = DataLoader(
    train_ds,
    batch_size=STAGE1_BATCH_SIZE,
    shuffle=True,
    num_workers=0  # Windows safe
)

val_loader = DataLoader(
    val_ds,
    batch_size=STAGE1_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ======================
# Model: EfficientNet-B3
# ======================
weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
model = models.efficientnet_b3(weights=weights)

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier (binary output)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 1)  # logits
)

model.to(device)

# ======================
# Loss & Optimizer
# ======================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=STAGE1_LR)

# ======================
# Training Loop
# ======================
best_val_loss = float("inf")

for epoch in range(STAGE1_EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(
        train_loader,
        desc=f"Stage-1 Epoch {epoch+1}/{STAGE1_EPOCHS}"
    ):
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # ------------------
    # Validation
    # ------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # ------------------
    # Save best model (SAFE)
    # ------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(os.path.dirname(STAGE1_MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), STAGE1_MODEL_PATH)
        print("✔ Saved best Stage-1 model")

print("✅ Stage-1 training complete")
