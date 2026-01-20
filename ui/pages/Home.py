import streamlit as st
from PIL import Image
import tempfile
import sys
import os

# --------------------------------------------------
# ADD PROJECT ROOT TO PATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
from src.inference import predict
from src.gradcam_stage2_swin import generate_gradcam

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Eye Disease Detection System")
st.caption("Multi-Stage AI-Based Ophthalmology Screening")

uploaded_file = st.file_uploader(
    "Upload an eye image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    result = predict(img_path)

    if result["stage"] == "Stage-1":
        st.success("🟢 NORMAL EYE")
        st.write(f"Probability: {result['probability']:.3f}")
    else:
        st.error("🔴 DISEASE DETECTED")
        st.write(f"Disease: {result['prediction']}")
        st.write(f"Confidence: {result['confidence']:.3f}")

        heatmap = generate_gradcam(img_path)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", width=350)
        with col2:
            st.image(heatmap, caption="Grad-CAM",  width=350)






