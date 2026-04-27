import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="Dental Diagnostic AI",
    page_icon="🦷",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0b1220;
        color: white;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    .stSelectbox label, .stFileUploader label, .stSubheader {
        color: white !important;
    }
    hr {
        border-color: #1e3a8a !important;
    }
    .badge-container {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    .ai-badge {
        background-color: #2563eb;
        color: white;
        padding: 6px 20px;
        border-radius: 999px;
        font-size: 16px;
        font-weight: bold;
        display: inline-block;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        color: #60a5fa;
        font-size: 14px;
        background-color: #0b1220;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    model_path = "best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

diagnoses = {
    "Caries": {"ar": "تسوس", "en": "Caries"},
    "Infection": {"ar": "عدوى", "en": "Infection"},
    "Fractured Teeth": {"ar": "كسر في السن", "en": "Fractured Teeth"},
    "Impacted teeth": {"ar": "سن مطمور", "en": "Impacted teeth"},
    "Healthy Teeth": {"ar": "سن سليم", "en": "Healthy Teeth"},
    "BDC-BDR": {"ar": "تغير في كثافة العظام", "en": "BDC-BDR"}
}

st.markdown("<h1 style='text-align: center; color: #60a5fa;'>🦷 Dental Diagnostic AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Clinical Decision Support System (CDSS)</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    conf_value = 0.50
    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])
    analyze_btn = st.button("Analyze Scan / تحليل", type="primary", use_container_width=True)

with col2:
    if analyze_btn and uploaded_file is None:
        st.error("⚠️ من فضلك قم برفع صورة الأشعة أولاً!")
    elif uploaded_file is not None:
        if analyze_btn:
            with st.spinner("Processing..."):
                if model is not None:
                    image = Image.open(uploaded_file).convert("RGB")
                    results = model.predict(source=image, conf=conf_value)[0]
                    res_plotted = results.plot()
                    
                    st.markdown("<div class='badge-container'><span class='ai-badge'>AI-Assisted Detection</span></div>", unsafe_allow_html=True)
                    st.image(res_plotted, channels="BGR", use_container_width=True, caption="Analysis Result")
                    
                    detected_classes = set([model.names[int(box.cls)] for box in results.boxes])
                    
                    st.subheader("📊 Detailed Clinical Report")
                    if not detected_classes:
                        st.info("No actionable findings detected.")
                    else:
                        for cls_name in detected_classes:
                            diag_info = diagnoses.get(cls_name, {"ar": cls_name, "en": cls_name})
                            st.success(f"**{diag_info['en']}** | {diag_info['ar']}")
                else:
                    st.error("Error: Model file 'best.pt' not found in repository.")
        else:
            st.image(uploaded_file, use_container_width=True, caption="Original X-Ray")
    else:
        st.info("Upload an X-Ray to view results")

st.markdown('<div class="footer">Developed By Eng. Hassan Masoud</div>', unsafe_allow_html=True)
