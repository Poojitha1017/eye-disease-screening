import torch
from torchvision import transforms, models
from timm import create_model
from PIL import Image

from src.config import *

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# STAGE 1 MODEL (EfficientNet-B3)
# Binary: Normal (1) vs Diseased (0)
# --------------------------------------------------
weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
stage1 = models.efficientnet_b3(weights=weights)

in_features = stage1.classifier[1].in_features
stage1.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features, 256),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 1)
)

stage1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device))
stage1.to(device).eval()

# --------------------------------------------------
# STAGE 2 MODELS
# --------------------------------------------------
swin = create_model(
    "swin_base_patch4_window7_224",
    pretrained=False,
    num_classes=NUM_STAGE2_CLASSES
)
swin.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=device))
swin.to(device).eval()

vit = create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=NUM_STAGE2_CLASSES
)
vit.load_state_dict(torch.load(STAGE2_VIT_MODEL_PATH, map_location=device))
vit.to(device).eval()

# --------------------------------------------------
# ENSEMBLE WEIGHTS
# --------------------------------------------------
SWIN_WEIGHT = 0.65
VIT_WEIGHT  = 0.35

# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------
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

# --------------------------------------------------
# INFERENCE (FINAL FIX)
# --------------------------------------------------
@torch.no_grad()
def predict(image_path):

    # ==============================
    # STAGE 1
    # ==============================
    img1 = preprocess(image_path, STAGE1_IMG_SIZE)
    stage1_prob = torch.sigmoid(stage1(img1)).item()

    # ------------------------------
    # HARD NORMAL GATE
    # ------------------------------
    NORMAL_THRESHOLD = 0.90  # very strict

    if stage1_prob >= NORMAL_THRESHOLD:
        return {
            "stage": "Stage-1",
            "prediction": "Normal",
            "stage1_probability": round(stage1_prob, 4)
        }

    # ==============================
    # STAGE 2 (only if NOT clearly normal)
    # ==============================
    img2 = preprocess(image_path, STAGE2_IMG_SIZE)

    swin_probs = torch.softmax(swin(img2), dim=1)
    vit_probs  = torch.softmax(vit(img2), dim=1)

    ensemble_probs = (
        SWIN_WEIGHT * swin_probs +
        VIT_WEIGHT  * vit_probs
    )

    cls_idx = torch.argmax(ensemble_probs, dim=1).item()
    confidence = ensemble_probs[0, cls_idx].item()

    # ------------------------------
    # LOW-CONFIDENCE REJECTION
    # ------------------------------
    if confidence < 0.60:
        return {
            "stage": "Stage-2",
            "prediction": "Normal (low disease confidence)",
            "stage1_probability": round(stage1_prob, 4),
            "confidence": round(confidence, 4)
        }

    return {
        "stage": "Stage-2 (Swin + ViT Ensemble)",
        "stage1_probability": round(stage1_prob, 4),
        "prediction": STAGE2_CLASS_NAMES[cls_idx],
        "confidence": round(confidence, 4),
        "class_probabilities": {
            STAGE2_CLASS_NAMES[i]: round(ensemble_probs[0, i].item(), 4)
            for i in range(NUM_STAGE2_CLASSES)
        }
    }









