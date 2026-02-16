from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch

from src.inference import predict
from src.gradcam_stage2_swin import generate_gradcam

app = FastAPI(title="Eye Disease Detection API")


@app.get("/")
def health():
    return {"status": "API is running"}


@app.post("/predict")
async def predict_eye_disease(file: UploadFile = File(...)):

    # ---------------- SAVE IMAGE ----------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        img_path = tmp.name

    # ---------------- INFERENCE ----------------
    result = predict(img_path)

    # ---------------- GRAD-CAM ----------------
    result["gradcam"] = None

    if "Stage-2" in result.get("stage", ""):
        with torch.enable_grad():
            heatmap = generate_gradcam(img_path)

        if isinstance(heatmap, np.ndarray):
            heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(heatmap)

            buffer = BytesIO()
            heatmap_pil.save(buffer, format="PNG")
            result["gradcam"] = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

    return result







