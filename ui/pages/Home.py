import streamlit as st
from PIL import Image
import tempfile
import requests
import base64
from io import BytesIO

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Eye Disease Detection",
    layout="centered"
)

st.title("👁️ Eye Disease Detection System")
st.caption("Multi-Stage AI-Based Ophthalmology Screening (Ensemble Model)")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an eye fundus image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    # --------------------------------------------------
    # SAVE TEMP IMAGE
    # --------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    # --------------------------------------------------
    # CALL FASTAPI
    # --------------------------------------------------
    with st.spinner("Running AI analysis..."):
        with open(img_path, "rb") as f:
            response = requests.post(
                API_URL,
                files={"file": f}
            )

        result = response.json()

    st.divider()

    # --------------------------------------------------
    # STAGE 1 RESULT
    # --------------------------------------------------
    if result["stage"] == "Stage-1":
        st.success("🟢 NORMAL EYE DETECTED")
        st.write(
            f"**Normal Confidence (Stage-1):** "
            f"`{result['stage1_probability']:.3f}`"
        )

    # --------------------------------------------------
    # STAGE 2 RESULT
    # --------------------------------------------------
    else:
        if result["prediction"].lower().startswith("normal"):
            st.warning("🟡 BORDERLINE CASE")
        else:
            st.error("🔴 DISEASE DETECTED")

        st.write(f"### 🩺 Diagnosis: **{result['prediction']}**")
        st.write(
            f"### 📊 Confidence: `{result.get('confidence', 0):.3f}`"
        )

        # --------------------------------------------------
        # GRAD-CAM DISPLAY
        # --------------------------------------------------
        if result.get("gradcam"):
            heatmap_bytes = base64.b64decode(result["gradcam"])
            heatmap_img = Image.open(BytesIO(heatmap_bytes))

            st.divider()
            st.subheader("🧠 Model Explainability (Grad-CAM)")

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", width=350)
            with col2:
                st.image(heatmap_img, caption="Grad-CAM", width=350)



