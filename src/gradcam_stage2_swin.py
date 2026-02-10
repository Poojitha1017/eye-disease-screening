import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from timm import create_model
from PIL import Image
import sys
import math

# ---------------- CONFIG ----------------
from src.config import (
    STAGE2_MODEL_PATH,
    STAGE2_IMG_SIZE,
    NUM_STAGE2_CLASSES,
    STAGE2_CLASS_NAMES
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------- MODEL ----------------
def load_model():
    model = create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,
        num_classes=NUM_STAGE2_CLASSES
    )
    model.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ---------------- IMAGE ----------------
transform = transforms.Compose([
    transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(DEVICE)
    return tensor, rgb

# ---------------- SWIN GRAD-CAM ----------------
class SwinGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output  # (B, N, C)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # (B, N, C)

    def generate(self, x):
        self.model.zero_grad()

        out = self.model(x)
        cls = out.argmax(dim=1).item()
        out[0, cls].backward()

        acts = self.activations[0]   # (N, C)
        grads = self.gradients[0]    # (N, C)

        weights = grads.mean(dim=0)  # (C,)
        cam = torch.matmul(acts, weights)  # (N,)

        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)

        # -------------------------------
        # 🔥 STRONG CAM NORMALIZATION
        # -------------------------------
        # Reshape tokens → spatial
        size = int(math.sqrt(cam.shape[0]))
        cam = cam.reshape(size, size)

        cam = cv2.resize(cam, (STAGE2_IMG_SIZE, STAGE2_IMG_SIZE))

        # Percentile clipping (huge difference)
        low, high = np.percentile(cam, 1), np.percentile(cam, 99)
        cam = np.clip(cam, low, high)

        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        # Gamma correction (ViT/Swin friendly)
        cam = np.power(cam, 0.5)

        # Smooth slightly (reduce patch artifacts)
        cam = cv2.GaussianBlur(cam, (7, 7), 0)

        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam, cls

# ---------------- OVERLAY ----------------
def overlay(img, cam):
    h, w = cam.shape
    img_resized = cv2.resize(img, (w, h))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Cleaner blend
    return cv2.addWeighted(img_resized, 0.55, heatmap, 0.6, 0)

# ======================================================
# ✅ API FUNCTION (FASTAPI / STREAMLIT SAFE)
# ======================================================
def generate_gradcam(img_path):
    model = load_model()

    target_layer = model.layers[-1].blocks[-1].norm2
    cam_engine = SwinGradCAM(model, target_layer)

    img_tensor, orig = load_image(img_path)

    with torch.enable_grad():
        cam, cls = cam_engine.generate(img_tensor)

    result = overlay(orig, cam)
    return result  # NumPy array (H, W, 3)

# ---------------- CLI MAIN ----------------
def main(img_path):
    model = load_model()

    target_layer = model.layers[-1].blocks[-1].norm2
    cam_engine = SwinGradCAM(model, target_layer)

    img_tensor, orig = load_image(img_path)

    with torch.enable_grad():
        cam, cls = cam_engine.generate(img_tensor)

    result = overlay(orig, cam)

    out_path = "gradcam_stage2_swin.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    print(f"Prediction: {STAGE2_CLASS_NAMES[cls]}")
    print(f"Grad-CAM saved to: {out_path}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.gradcam_stage2_swin <image_path>")
        sys.exit(1)

    main(sys.argv[1])
