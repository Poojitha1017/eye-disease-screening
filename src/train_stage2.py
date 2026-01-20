# train_stage2.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from tqdm import tqdm
import random
import numpy as np

from config import *

# ----------------------
# Reproducibility
# ----------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------
# Device
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------
# Transforms
# ----------------------
train_transform = transforms.Compose([
    transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
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

# ----------------------
# Datasets & Loaders
# ----------------------
train_ds = datasets.ImageFolder(STAGE2_TRAIN_DIR, transform=train_transform)
val_ds   = datasets.ImageFolder(STAGE2_VAL_DIR, transform=val_transform)

print("Stage-2 classes:", train_ds.classes)

train_loader = DataLoader(
    train_ds,
    batch_size=max(1, STAGE2_BATCH_SIZE//2),  # reduced for CPU/laptop
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=max(1, STAGE2_BATCH_SIZE//2),
    shuffle=False,
    num_workers=0
)

# ----------------------
# Model: Swin Transformer (no CBAM)
# ----------------------
num_classes = len(train_ds.classes)
model = create_model(
    "swin_base_patch4_window7_224",
    pretrained=True,
    num_classes=num_classes
)
model.to(device)

# ----------------------
# Loss & Optimizer
# ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=STAGE2_LR)

# ----------------------
# Training Loop
# ----------------------
best_val_loss = float("inf")

for epoch in range(STAGE2_EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Stage-2 Epoch {epoch+1}/{STAGE2_EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # ----------------------
    # Validation
    # ----------------------
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

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ----------------------
    # Save best model
    # ----------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), STAGE2_MODEL_PATH)
        print("✔ Saved best Stage-2 model")

print("✅ Stage-2 training complete")
