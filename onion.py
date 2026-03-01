import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================================
# 1. Page Configuration & Professional Styling
# ==========================================
st.set_page_config(
    page_title="OnionGuard AI",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a modern, attractive UI
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }

    /* Sidebar styling with improved visibility */
    section[data-testid="stSidebar"] {
        background-color: #1b5e20 !important;
    }

    /* Make all sidebar text white */
    section[data-testid="stSidebar"] .stText, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }

    /* Specifically fix Metric colors for visibility */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
    }

    /* Header styling */
    .main-title {
        font-size: 45px;
        font-weight: 800;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 10px;
    }

    
    /* Prediction Card styling */
    .res-card {
        background-color: black;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-left: 10px solid #2e7d32;
        color: #ffffff; /* This makes the text pure white */
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        height: 3.5em;
        background: linear-gradient(45deg, #2e7d32, #43a047);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(46, 125, 50, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Sidebar - Project Metrics & Info
# ==========================================
with st.sidebar:
    st.image("update_profile.jpg", width=500)
    st.title("Project Dashboard")
    st.markdown("---")

    st.subheader("📊 Model Metrics")
    # Metric values are now styled white in the CSS above
    st.metric(label="Model Accuracy", value="96.09%")
    st.metric(label="Classes Identified", value="11")

    st.markdown("---")
    st.subheader("📂 Project Info")
    st.write("**Student:** Atharv Taral")
    # Updated Domain as per your request
    st.write("**Domain:** Machine Learning")
    # Refined Focus for a professional look
    st.write("**Focus:** Intelligent Plant Pathology in Smart Agriculture")

    # Removed the info box part as requested


# ==========================================
# 3. Model Loading
# ==========================================
@st.cache_resource
def load_my_model():
    path = "best_model.h5"
    return tf.keras.models.load_model(path)


try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

class_labels = [
    'Alternaria_D', 'Botrytis Leaf Blight', 'Bulb_blight-D', 'Caterpillar-P',
    'Fusarium-D', 'Healthy leaves', 'Iris yellow virus_augment', 'Rust',
    'Virosis-D', 'Xanthomonas Leaf Blight', 'stemphylium Leaf Blight'
]

# ==========================================
# 4. Main Interface
# ==========================================
st.markdown("<h1 class='main-title'>🧅 OnionGuard: Disease Diagnostic AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Secure your crop with instant, AI-powered leaf analysis</p>",
            unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Section")
    uploaded_file = st.file_uploader("Select an image of the onion leaf", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        st.image(img, caption='Preview', use_container_width=True)

with col2:
    st.subheader("🔍 Analysis & Diagnosis")
    if uploaded_file is not None:
        img_resized = img.resize((100, 100))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        if st.button('✨ Start AI Diagnosis'):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)

            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions)
            result = class_labels[class_idx]
            confidence = np.max(predictions) * 100

            st.markdown(f"""
                <div class="res-card">
                    <h2 style='color: #2e7d32;'>Diagnosis Complete</h2>
                    <p style='font-size: 20px;'><b>Detected:</b> {result}</p>
                    <p style='font-size: 18px;'><b>Confidence:</b> {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            if result == 'Healthy leaves':
                st.balloons()
                st.success("The crop appears to be in excellent health!")
            else:
                st.error("Immediate attention may be required for the detected disease.")
    else:
        st.info("Please upload an image to enable diagnostic features.")

# ==========================================
# 5. Footer
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: red;'>© 2026 Atharv Taral | Final Year Project | Savitribai Phule Pune University</p>",
    unsafe_allow_html=True
)
