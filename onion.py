import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os
import openai
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# ==========================================
# 1. Configuration & Session State Setup
# ==========================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("❌ OpenAI API Key not found in .env file!")
    st.stop()

client = openai.OpenAI(api_key=API_KEY)

# --- Initialize Session States ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "diagnosis_done" not in st.session_state:
    st.session_state.diagnosis_done = False
if "detected_result" not in st.session_state:
    st.session_state.detected_result = None
if "advice" not in st.session_state:
    st.session_state.advice = None
if "med_img" not in st.session_state:
    st.session_state.med_img = None
if "main_audio" not in st.session_state:
    st.session_state.main_audio = None

# ==========================================
# 2. Expert AI Advisor Functions
# ==========================================
def get_expert_advice(disease_name, user_query="What is the treatment?"):
    prompt = f"""
    You are an expert Agriculture Scientist. The onion crop has '{disease_name}'.
    Provide the following details in simple bullet points:
    - Cause of the disease.
    - Recommended Medicine/Fungicide (Exact name for Google search).
    - Application method (How to spray).
    - Safety precautions for the farmer.
    Farmer's Question: {user_query}
    Keep it professional but easy to understand.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful Agri-Expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def text_to_speech(text):
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text[:4000]) # Limit chars for safety
    audio_path = f"temp_audio_{int(time.time())}.mp3"
    response.stream_to_file(audio_path)
    return audio_path

# ==========================================
# 3. GLOBAL PREMIUM STYLING
# ==========================================
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa !important; }
    section[data-testid="stSidebar"] { background-color: #1b5e20 !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    .hero-section { text-align: center; padding: 20px; margin-top: -30px; }
    .hero-title { font-size: 50px !important; font-weight: 900 !important; color: #1b5e20 !important; }
    .res-card { background-color: white; padding: 20px; border-radius: 15px; border-left: 10px solid #1b5e20; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. Sidebar & Model Loading
# ==========================================
with st.sidebar:
    st.image("update_profile.jpg", width=150) # Make sure this file exists
    st.title("Project Dashboard")
    st.metric(label="Model Accuracy", value="96.09%")
    st.write("**Student:** Atharv Taral")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_my_model()

class_labels = [
    'Alternaria_D', 'Botrytis Leaf Blight', 'Bulb_blight-D', 'Caterpillar-P',
    'Fusarium-D', 'Healthy leaves', 'Iris yellow virus_augment', 'Rust',
    'Virosis-D', 'Xanthomonas Leaf Blight', 'stemphylium Leaf Blight'
]

# ==========================================
# 5. Main UI Logic
# ==========================================
st.markdown('<div class="hero-section"><div class="hero-title">🧅 OnionGuard AI</div></div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Leaf Scan")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_container_width=True)

with col2:
    st.subheader("🔍 Diagnosis")
    if uploaded_file:
        if st.button('✨ Start AI Diagnosis'):
            # Preprocessing
            img_resized = img.resize((100, 100))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            with st.spinner("Analyzing..."):
                predictions = model.predict(img_array)
                class_idx = np.argmax(predictions)
                st.session_state.detected_result = class_labels[class_idx]
                st.session_state.diagnosis_done = True
                
                if st.session_state.detected_result != 'Healthy leaves':
                    advice = get_expert_advice(st.session_state.detected_result)
                    st.session_state.advice = advice
                    st.session_state.main_audio = text_to_speech(advice)
                
            st.balloons()

    if st.session_state.diagnosis_done:
        st.markdown(f"""
            <div class="res-card">
                <h3>Result: {st.session_state.detected_result}</h3>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# 6. Treatment & Interactive Chat (The Core Fix)
# ==========================================
if st.session_state.diagnosis_done and st.session_state.detected_result != 'Healthy leaves':
    st.markdown("---")
    st.subheader("📋 Expert Advice & Audio")
    st.write(st.session_state.advice)
    st.audio(st.session_state.main_audio)

    st.markdown("---")
    st.subheader("💬 Chat with Krishi-Mitra AI")

    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["q"])
        with st.chat_message("assistant"):
            st.write(chat["a"])

    # Chat Input
    user_query = st.chat_input("Ask a follow-up question...")
    
    # Voice Input
    st.write("🎤 **Or ask via voice:**")
    voice_data = mic_recorder(start_prompt="Record", stop_prompt="Stop", key="recorder")

    query_to_process = None
    if user_query:
        query_to_process = user_query
    elif voice_data:
        with open("temp_v.wav", "wb") as f:
            f.write(voice_data['bytes'])
        transcript = client.audio.transcriptions.create(model="whisper-1", file=open("temp_v.wav", "rb"))
        query_to_process = transcript.text

    if query_to_process:
        with st.spinner("Krishi-Mitra is thinking..."):
            new_advice = get_expert_advice(st.session_state.detected_result, query_to_process)
            # Add to history
            st.session_state.chat_history.append({"q": query_to_process, "a": new_advice})
            # Rerun to show new message
            st.rerun()

st.markdown("<br><hr><p style='text-align:center;'>© 2026 Atharv Taral | SPPU</p>", unsafe_allow_html=True)
