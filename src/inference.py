import torch
import cv2
import numpy as np
from torchvision import transforms, models
from timm import create_model
from PIL import Image

from src.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Stage-1 Model (EfficientNet-B3)
# -------------------------------
weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
stage1 = models.efficientnet_b3(weights=weights)

in_features = stage1.classifier[1].in_features
stage1.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 1)
)

stage1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device))
stage1.to(device).eval()

# -------------------------------
# Stage-2 Model (Swin Transformer)
# -------------------------------
stage2 = create_model(
    "swin_base_patch4_window7_224",
    pretrained=False,
    num_classes=NUM_STAGE2_CLASSES
)

stage2.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=device))
stage2.to(device).eval()

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess(img_path, img_size):
    img = Image.open(img_path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return tfm(img).unsqueeze(0).to(device)

# -------------------------------
# Prediction Pipeline
# -------------------------------
@torch.no_grad()
def predict(image_path):
    img1 = preprocess(image_path, STAGE1_IMG_SIZE)
    prob = torch.sigmoid(stage1(img1)).item()

    if prob >= STAGE1_THRESHOLD:
        return {
            "stage": "Stage-1",
            "prediction": "NORMAL",
            "probability": prob
        }

    img2 = preprocess(image_path, STAGE2_IMG_SIZE)
    logits = stage2(img2)
    probs = torch.softmax(logits, dim=1)
    cls = torch.argmax(probs, dim=1).item()

    return {
        "stage": "Stage-2",
        "prediction": STAGE2_CLASS_NAMES[cls],
        "confidence": probs[0, cls].item()
    }




