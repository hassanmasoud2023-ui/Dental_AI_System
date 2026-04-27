import streamlit as st
import numpy as np
from PIL import Image
import os

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model_path = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
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

st.set_page_config(
    page_title="Dental Diagnostic AI",
    page_icon="🦷",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0b1220 0%, #1e293b 100%);
    color: white;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
.stSelectbox label, .stFileUploader label, .stSubheader, .stButton > div > div {
    color: white !important;
}
hr {
    border-color: #1e3a8a !important;
}
.badge-container {
    display: flex;
    justify-content: center;
    margin: 20px 0;
}
.ai-badge {
    background: linear-gradient(45deg, #2563eb, #3b82f6);
    color: white;
    padding: 8px 24px;
    border-radius: 25px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 15px;
    color: #60a5fa;
    font-size: 14px;
    background: rgba(11, 18, 32, 0.95);
    z-index: 1000;
}
.metric-card {
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h1 style='color: #60a5fa; font-size: 3rem; margin: 0;'>🦷 Dental Diagnostic AI</h1>
    <p style='color: #94a3b8; font-size: 1.2rem; margin: 0;'>Clinical Decision Support System (CDSS) | نظام دعم القرار السريري</p>
</div>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.markdown("### ⚙️ Settings")
    
    conf_value = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.50, 
        step=0.05
    )
    
    uploaded_file = st.file_uploader(
        "📁 Upload X-Ray Image", 
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    analyze_btn = st.button(
        "🚀 Analyze Scan / تحليل الأشعة", 
        type="primary", 
        use_container_width=True
    )

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="🖼️ Original X-Ray")
        
        if analyze_btn:
            if model is not None:
                with st.spinner("🔬 AI is analyzing your X-Ray..."):
                    try:
                        img_array = np.array(image)
                        results = model.predict(
                            source=img_array, 
                            conf=conf_value,
                            verbose=False
                        )[0]
                        
                        res_plotted = results.plot()
                        st.markdown("""
                        <div class='badge-container'>
                            <span class='ai-badge'>✅ AI Analysis Complete</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.image(res_plotted, channels="BGR", use_container_width=True, caption="📊 AI Detection Results")
                        
                        detected_classes = []
                        if results.boxes is not None:
                            for box in results.boxes:
                                cls_id = int(box.cls)
                                cls_name = model.names[cls_id]
                                confidence = float(box.conf)
                                detected_classes.append((cls_name, confidence))
                        
                        st.markdown("### 📋 Clinical Report")
                        st.markdown("---")
                        
                        if not detected_classes:
                            st.info("✅ **No pathological findings detected.**")
                            st.success("🟢 Healthy teeth structure")
                        else:
                            for cls_name, confidence in detected_classes:
                                diag_info = diagnoses.get(cls_name, {"ar": cls_name, "en": cls_name})
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h3 style='color: #10b981; margin: 0 0 10px 0;'>{diag_info['en']}</h3>
                                    <p style='color: #94a3b8; margin: 0 0 5px 0;'>{diag_info['ar']}</p>
                                    <p style='color: #60a5fa; font-size: 1.1rem; margin: 0;'>
                                        Confidence: <strong>{confidence:.1%}</strong>
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("💡 Try lowering confidence threshold or check image format")
            else:
                st.warning("⚠️ Model not loaded. Please refresh the page.")
    elif analyze_btn:
        st.error("❌ Please upload an X-Ray image first!")
    else:
        st.info(
            "👆 Upload your X-Ray image and click **Analyze** to get AI-powered diagnosis.\n\n"
            "✅ Supports JPG, PNG, BMP formats\n"
            "⚡ Results in seconds"
        )

st.markdown("""
<div class="footer">
    🦷 Developed by Eng. Hassan Masoud | Powered by YOLOv8 & Streamlit
</div>
""", unsafe_allow_html=True)
