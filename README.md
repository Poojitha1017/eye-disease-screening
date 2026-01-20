# 👁️ Eye Disease Detection System

An AI-powered eye disease detection system that analyzes retinal images to identify the presence of ocular diseases using deep learning and explainable AI techniques.

---

## 🚀 Overview

This project presents an AI-based eye disease detection application designed to determine whether an eye disease is present in retinal images.

The system performs:
- Automated disease detection from eye images
- Confidence-aware predictions
- Visual explainability using Grad-CAM to highlight image regions influencing the model’s decision

The application is implemented as an interactive web interface using **Streamlit**, enabling easy image upload and real-time inference.

---

## ✨ Key Features

- 📤 **Simple Image Upload**  
  Upload retinal images directly through a web interface.

- 🧠 **Multi-Stage Disease Detection**  
  Robust deep learning pipeline for disease presence detection.

- 📊 **Detailed Analysis**  
  Provides disease classification results along with confidence scores.

- 🔍 **Explainable AI (Grad-CAM)**  
  Highlights regions of the image contributing to the disease prediction for better interpretability.

- ⚡ **Real-Time Inference**  
  Fast predictions optimized for demos and practical usage.

---

## 🛠️ Technologies Used

- Python  
- PyTorch  
- EfficientNet  
- Swin Transformer  
- Grad-CAM  
- OpenCV  
- Streamlit  

---

## 🧩 Project Structure

```bash
Eye-Disease-Detection/
│
├── src/
│   ├── inference.py
│   ├── gradcam_stage2_swin.py
│   ├── config.py
│
├── ui/
│   └── app.py
│
├── models/
│   └── (trained model weights)
│
├── requirements.txt
├── README.md
└── .gitignore


▶️ How to Run the Application

▶️ How to Run the Application
   
   git clone https://github.com/your-username/eye-disease-detection.git
   cd eye-disease-detection

2️⃣ Create a Virtual Environment (Recommended)
    
    python -m venv venv
    source venv/bin/activate      # Linux / Mac
    venv\Scripts\activate         # Windows

3️⃣ Install Dependencies
  
    streamlit run ui/app.py

## ⚠️ Disclaimer

This tool is intended for academic and research purposes only.

It does not provide medical diagnosis and should not be used as a substitute for professional ophthalmological evaluation.
All predictions should be verified by a qualified medical professional.


--

## 📌 Notes

The project focuses only on disease detection, not diagnosis or treatment.

--
