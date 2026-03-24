import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os
import openai
import requests
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# ==========================================
# 1. Configuration & API Setup
# ==========================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("❌ OpenAI API Key not found in .env file!")
    st.stop()

client = openai.OpenAI(api_key=API_KEY)

# ==========================================
# 2. UI Styling (Premium Agriculture Theme)
# ==========================================
st.set_page_config(page_title="OnionGuard AI", page_icon="🧅", layout="wide")

st.markdown("""
    <style>
    /* Global Styles */
    .stApp { background-color: #fcfdfc; }
    
    /* Main Title Styling */
    .main-title { 
        font-size: 48px !important; 
        font-weight: 900 !important; 
        color: #1b5e20 !important; 
        text-align: center !important; 
        margin-top: -30px !important;
        text-shadow: 2px 2px 4px #d1d1d1;
    }
    
    /* Section Headers */
    h3, .stMarkdown h3 { 
        color: #2e7d32 !important; 
        font-weight: 700 !important;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
    }

    /* AI Advice & Bullet Points - High Visibility */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #262730 !important; 
        font-size: 18px !important;
        line-height: 1.7 !important;
    }

    /* Diagnosis Result Card (Modern Dark) */
    .res-card { 
        background: linear-gradient(135deg, #121212 0%, #1a1a1a 100%);
        padding: 30px; 
        border-radius: 20px; 
        border-left: 12px solid #43a047; 
        box-shadow: 0px 10px 20px rgba(0,0,0,0.2);
        margin-bottom: 25px;
        color: white !important;
    }
    .res-card h2 { color: #ffffff !important; margin-bottom: 10px; }
    .res-card p { color: #e0e0e0 !important; font-size: 19px !important; }

    /* Custom Info Box */
    .stAlert {
        background-color: #e8f5e9 !important;
        border: 1px solid #c8e6c9 !important;
        border-radius: 10px;
    }
    .stAlert p { color: #1b5e20 !important; font-weight: 600 !important; }

    /* Image Shadow effect */
    img { border-radius: 15px; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] { background-color: #1b5e20 !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. Core Functions (AI & Logic)
# ==========================================

def get_expert_advice(disease_name, user_query="What is the treatment?"):
    prompt = f"""
    You are an expert Agriculture Scientist. The onion crop has '{disease_name}'.
    Provide the following details in simple bullet points:
    - Cause: Briefly explain the biological reason.
    - Recommended Medicine: Give the exact commercial fungicide name.
    - Application Method: Specific instructions on dosage/spray.
    - Farmer Safety: Key protective measures.
    
    Farmer's Question: {user_query}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a professional Agri-Expert."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_medicine_image(query):
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CX")
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {'q': query + " fungicide agriculture medicine bottle", 'cx': cx, 'key': api_key, 'searchType': 'image', 'num': 1}
    try:
        response = requests.get(search_url, params=params)
        data = response.json()
        return data['items'][0]['link'] if 'items' in data else None
    except: return None

def text_to_speech(text):
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text[:400]) # limit to 400 chars for speed
    audio_path = "crop_advice.mp3"
    response.stream_to_file(audio_path)
    return audio_path

# ==========================================
# 4. Sidebar & Model
# ==========================================
with st.sidebar:
    st.image("update_profile.jpg", width=200)
    st.markdown("### Project Dashboard")
    st.metric(label="Model Accuracy", value="96.09%")
    st.markdown("---")
    st.write("**Student:** Atharv Taral")
    st.write("**Domain:** Agentic AI in Agriculture")
    st.write("**University:** SPPU, Pune")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_my_model()
class_labels = ['Alternaria_D', 'Botrytis Leaf Blight', 'Bulb_blight-D', 'Caterpillar-P', 
                'Fusarium-D', 'Healthy leaves', 'Iris yellow virus', 'Rust', 
                'Virosis-D', 'Xanthomonas Leaf Blight', 'stemphylium Leaf Blight']

# ==========================================
# 5. Main Content
# ==========================================
st.markdown("<h1 class='main-title'>🧅 OnionGuard: Agentic Diagnostic AI</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 📤 Upload Leaf Scan")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_container_width=True)

with col2:
    st.markdown("### 🔍 Analysis & Agentic Advice")
    if uploaded_file:
        img_resized = img.resize((100, 100))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button('✨ Launch Diagnostic Agent'):
            progress_bar = st.progress(0)
            for p in range(100):
                time.sleep(0.005)
                progress_bar.progress(p + 1)

            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions)
            result = class_labels[class_idx]
            confidence = np.max(predictions) * 100

            st.markdown(f"""
                <div class="res-card">
                    <h2>Diagnosis Complete</h2>
                    <p><b>Condition Identified:</b> {result}</p>
                    <p><b>Prediction Confidence:</b> {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            if result != 'Healthy leaves':
                with st.spinner("🤖 Consulting AI Expert & Fetching Medicine Data..."):
                    advice = get_expert_advice(result)
                    st.session_state.advice = advice
                    st.session_state.detected_result = result
                    st.session_state.med_img = get_medicine_image(result)
                    st.session_state.audio = text_to_speech(advice)
            else:
                st.balloons()
                st.success("Your onion crop is healthy! No treatment needed.")

    # Display AI Expert Results
    if "advice" in st.session_state and uploaded_file:
        st.markdown("---")
        st.markdown("### 📋 Treatment Expert Suggestion")
        
        # Medicine Photo
        if st.session_state.med_img:
            st.image(st.session_state.med_img, caption=f"Recommended Product for {st.session_state.detected_result}", width=350)
        else:
            st.info("ℹ️ Direct medicine reference photo not available. Please consult the text below.")

        st.write(st.session_state.advice)
        st.audio(st.session_state.audio)

        # Voice Chat Section
        st.markdown("---")
        st.subheader("💬 Chat with Krishi-Mitra AI")
        
        voice_data = mic_recorder(start_prompt="🎤 Ask a Question (Voice)", stop_prompt="🛑 Stop", key="recorder")
        user_text = st.chat_input("Or type your question here...")

        query = None
        if user_text: query = user_text
        elif voice_data:
            with open("temp_v.wav", "wb") as f: f.write(voice_data['bytes'])
            transcript = client.audio.transcriptions.create(model="whisper-1", file=open("temp_v.wav", "rb"))
            query = transcript.text

        if query:
            with st.spinner("Thinking..."):
                new_advice = get_expert_advice(st.session_state.detected_result, query)
                st.info(f"**Question:** {query}\n\n**AI Expert:** {new_advice}")
                st.audio(text_to_speech(new_advice))

# ==========================================
# 6. Footer
# ==========================================
st.markdown("<br><br><hr><p style='text-align:center; color:#777;'>Developed by Atharv Taral | Savitribai Phule Pune University | 2026</p>", unsafe_allow_html=True)
