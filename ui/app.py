import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Eye Disease Detection",
    layout="wide",
)

# --------------------------------------------------
# CUSTOM CSS (CARD UI)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .card {
        background-color: #0f1117;
        border: 1px solid #2c2f36;
        border-radius: 16px;
        padding: 24px;
        height: 100%;
    }
    .card h3 {
        margin-bottom: 10px;
        font-size: 22px;
    }
    .card p {
        color: #c7c7c7;
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("👁️ Eye Disease Detection System")
st.markdown(
    "AI-powered eye disease screening using deep learning and explainable AI."
)

st.markdown("---")

# --------------------------------------------------
# FEATURE CARDS
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <h3>⬆️ Easy Image Upload</h3>
            <p>
                Simple and secure retinal image upload designed for
                rapid eye disease screening in clinical environments.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <h3>📋 Detailed Analysis</h3>
            <p>
                Comprehensive reports including disease classification
                and confidence scores to assist clinical decision-making.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <h3>📊 Explainable Disease Highlighting</h3>
            <p>
                Visually highlights regions influencing the model’s
                prediction, improving transparency and clinical interpretability.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------
st.info(
    "⚠️ **Disclaimer:** This system is intended to assist clinical screening "
    "and decision support. Final diagnosis and treatment decisions should "
    "always be confirmed by a qualified ophthalmologist."
)



