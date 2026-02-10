# test_stage2_vit.py
import torch
import argparse
from torchvision import transforms
from timm import create_model
from PIL import Image

from src.config import *

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# Argument parser
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    type=str,
    required=True,
    help="Path to input image"
)
args = parser.parse_args()

# --------------------------------------------------
# Transforms (same as validation)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Load image
# --------------------------------------------------
img = Image.open(args.image).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# --------------------------------------------------
# Load ViT Model
# --------------------------------------------------
model = create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=len(STAGE2_CLASS_NAMES)
)

model.load_state_dict(
    torch.load(STAGE2_VIT_MODEL_PATH, map_location=device)
)
model.to(device)
model.eval()

# --------------------------------------------------
# Inference
# --------------------------------------------------
with torch.no_grad():
    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)
    conf, pred_idx = torch.max(probs, dim=1)

pred_class = STAGE2_CLASS_NAMES[pred_idx.item()]
confidence = conf.item()

# --------------------------------------------------
# Result
# --------------------------------------------------
print("\n🧠 ViT Stage-2 Prediction")
print("----------------------------")
print(f"Prediction : {pred_class}")
print(f"Confidence : {confidence:.4f}")


