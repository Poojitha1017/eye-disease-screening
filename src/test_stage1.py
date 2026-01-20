# src/test_stage1.py
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from config import *

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((STAGE1_IMG_SIZE, STAGE1_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# Load Model
# -----------------------
weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
model = models.efficientnet_b3(weights=weights)

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 1)
)

model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------
# Read Image Path
# -----------------------
if len(sys.argv) != 2:
    print("Usage: python src/test_stage1.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]

image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# -----------------------
# Inference
# -----------------------
with torch.no_grad():
    logits = model(image)
    prob = torch.sigmoid(logits).item()

prediction = "Normal" if prob >= STAGE1_THRESHOLD else "Diseased"

print(f"Probability (Diseased): {prob:.4f}")
print(f"Prediction: {prediction}")
