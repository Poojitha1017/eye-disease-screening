# test_stage2.py
import torch
import cv2
from torchvision import transforms
from timm import create_model
from PIL import Image
from config import *

# ------------------------------
# Device
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Load model (Stage-2 Swin)
# ------------------------------
model = create_model(
    "swin_base_patch4_window7_224",
    pretrained=False,
    num_classes=NUM_STAGE2_CLASSES
)

model.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ------------------------------
# Preprocessing
# ------------------------------
preprocess = transforms.Compose([
    transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# Load image
# ------------------------------
def load_image(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    return tensor

# ------------------------------
# Main inference
# ------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python test_stage2.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    image_tensor = load_image(img_path)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, class_idx].item()

    pred_class = STAGE2_CLASS_NAMES[class_idx]

    print(f"Prediction : {pred_class}")
    print(f"Confidence : {confidence:.4f}")
