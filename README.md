## 👁️ Eye Disease Detection System (Ensemble + Explainability)

A production-ready deep learning system for automated eye disease detection using an ensemble of Swin Transformer and Vision Transformer (ViT), enhanced with Grad-CAM explainability for medical interpretability.

-- 

## 🚀 Key Features

✅ Two-stage disease detection pipeline

🧠 Ensemble learning (Swin Transformer + ViT) using soft-weighted probabilities

🔍 Grad-CAM explainability (Swin as primary explainable model)

-- 

## 🩺 Diseases Supported

Diabetic Retinopathy (DR)

Cataract

Conjunctivitis

Each prediction includes:

Final class label

Confidence score

Per-class probabilities

Grad-CAM heatmap (visual explanation)

-- 

## 🧠 Architecture Overview
🔹 Stage 1 (Binary Screening)

Filters images that require deeper analysis

🔹 Stage 2 (Ensemble Classification)

Swin Transformer → primary model

Vision Transformer (ViT) → secondary model

Soft-weighted ensemble combines probabilities

📌 Important Design Choice

Grad-CAM is generated only from Swin Transformer

Reason: Swin provides spatially meaningful attention maps

Ensemble is used for prediction accuracy, not explainability

This is a recommended and accepted practice in medical AI.

-- 

## 🗂️ Project Structure

Eye Disease Detection/
│
├── api/
│   └── app.py                  # FastAPI entry point
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── gradcam_stage2_swin.py
│   ├── inference.py
│   ├── utils.py
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── train_stage2_vit.py
│   ├── test_stage1.py
│   ├── test_stage2.py
│   └── test_stage2_vit.py
│
├── models/
│   └── stage2_swin.pth
│
├── ui/
│   └── pages/
│       └── Home.py
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md


## ⚙️ Setup & Installation (Local)

# 1️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows

# 2️⃣ Install dependencies
pip install -r requirements.txt

# ▶️ Running the Application
🔹 FastAPI Backend
   uvicorn api.app:app --reload
   
   Open Swagger UI:
   http://localhost:8000/docs
  
## 🐳 Docker Deployment
   
   # Build the image
    docker build -t eye-disease-api .

  # Run the container
    docker run -p 8000:8000 eye-disease-api

  Open:
  http://localhost:8000/docs




## ⚠️ Disclaimer

This tool is intended for academic and research purposes only.

It does not provide medical diagnosis and should not be used as a substitute for professional ophthalmological evaluation.
All predictions should be verified by a qualified medical professional.


--

## 📌 Notes

The project focuses only on disease detection, not diagnosis or treatment.

--
