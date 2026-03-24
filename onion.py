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
# 1. Configuration & API Setup
# ==========================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("❌ OpenAI API Key not found in .env file!")
    st.stop()

client = openai.OpenAI(api_key=API_KEY)

# ==========================================
# 2. Expert AI Advisor Functions
# ==========================================

def get_expert_advice(disease_name, user_query="What is the treatment?"):
    """Fetch medicine and treatment advice from GPT-4o"""
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
        messages=[{"role": "system", "content": "You are a helpful Agri-Expert."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


import requests
def get_medicine_image(query):
    """Searching for a medicine photo from Google"""
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CX")
    search_url = "https://www.googleapis.com/customsearch/v1"

    params = {
        'q': query + " agriculture medicine fungicide bottle",
        'cx': cx,
        'key': api_key,
        'searchType': 'image',
        'num': 1
    }
    try:
        response = requests.get(search_url, params=params)
        data = response.json()
        if 'items' in data:
            return data['items'][0]['link']
        else:
            print(f"Google Search Response: {data}")
    except Exception as e:
        print(f"Error fetching image: {e}")

def text_to_speech(text):
    """Convert AI advice to Audio speech"""
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    audio_path = "crop_advice.mp3"
    response.stream_to_file(audio_path)
    return audio_path

# ==========================================
# 3. Page Layout & Styling (Fixed Visibility)
# ==========================================
st.set_page_config(page_title="OnionGuard AI", page_icon="🧅", layout="wide")

st.markdown("""
    <style>
    /* 1. Overall App Background */
    .stApp { 
        background-color: #fcfdfc !important; 
    }
    
    /* FIX: Main Title Visibility - Bold Dark Green */
    .main-title { 
        font-size: 42px !important; 
        font-weight: 850 !important; 
        color: #1b5e20 !important; 
        text-align: center !important; 
        margin-bottom: 30px !important;
        display: block !important;
    }
    
    /* 2. Subheaders Visibility */
    h3, .stMarkdown h3 { 
        color: #2e7d32 !important; 
        font-weight: 700 !important;
        font-size: 26px !important;
    }

    /* 3. AI Advice Text (Dark Charcoal for readability) */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #262730 !important; 
        font-size: 18px !important;
        line-height: 1.6 !important;
    }

    /* 4. Info Box */
    .stAlert {
        background-color: #e8f5e9 !important;
        border: 1px solid #c8e6c9 !important;
    }
    .stAlert p {
        color: #1b5e20 !important;
        font-weight: 500;
    }

    /* 5. Diagnosis Result Card */
    .res-card { 
        background-color: #121212 !important; 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 8px solid #43a047; 
        margin-bottom: 20px;
    }
    .res-card h2, .res-card p {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. Sidebar & Model Loading
# ==========================================
with st.sidebar:
    st.image("update_profile.jpg", width=200)
    st.title("Project Dashboard")
    st.metric(label="Model Accuracy", value="96.09%")
    st.write("**Student:** Atharv Taral")
    st.write("**Domain:** Smart Agriculture AI")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("best_model.h5")

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
# 5. Main Dashboard
# ==========================================
st.markdown("<h1 class='main-title'>🧅 OnionGuard: Agentic Diagnostic AI</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Leaf Scan")
    uploaded_file = st.file_uploader("Select an image of the onion leaf", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Preview', use_container_width=True)

with col2:
    st.subheader("🔍 Analysis & Expert Advice")
    if uploaded_file:
        img_resized = img.resize((100, 100))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button('✨ Start AI Diagnosis'):
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent + 1)

            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions)
            result = class_labels[class_idx]
            confidence = np.max(predictions) * 100

            st.markdown(f"""
                <div class="res-card">
                    <h2 style='color: white;'>Diagnosis Complete</h2>
                    <p style='font-size: 20px;'><b>Detected Disease:</b> {result}</p>
                    <p style='font-size: 18px;'><b>Confidence:</b> {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            if result != 'Healthy leaves':
                with st.spinner("🤖 Consulting AI Expert & Finding Medicine..."):
                    advice = get_expert_advice(result)
                    st.session_state.advice = advice
                    st.session_state.detected_result = result
                    med_img = get_medicine_image(result)
                    st.session_state.med_img = med_img
                    audio_path = text_to_speech(advice)
                    st.session_state.audio = audio_path
            else:
                st.balloons()
                st.success("The crop appears to be in excellent health!")

            if "advice" in st.session_state:
                st.markdown("---")
                st.markdown("### 📋 Treatment Expert Suggestion")

                if "med_img" in st.session_state and st.session_state.med_img:
                    st.image(st.session_state.med_img,
                             caption=f"Recommended Product for {st.session_state.detected_result}",
                             width=300)
                else:
                    st.info("ℹ️ Looking for a reference photo of the medicine or it is not available.")

                st.write(st.session_state.advice)
                st.audio(st.session_state.audio)

                st.markdown("---")
                st.subheader("💬 Chat with Krishi-Mitra AI")

            user_text_query = st.chat_input("Type your question here...")
            st.write("🎤 **Or ask out loud:**")
            voice_data = mic_recorder(start_prompt="Record Question", stop_prompt="Stop Recording", key="recorder")

            query_to_process = None
            if user_text_query:
                query_to_process = user_text_query
            elif voice_data:
                with open("temp_v.wav", "wb") as f:
                    f.write(voice_data['bytes'])
                with st.spinner("Transcribing voice..."):
                    transcript = client.audio.transcriptions.create(model="whisper-1", file=open("temp_v.wav", "rb"))
                    query_to_process = transcript.text

            if query_to_process:
                with st.spinner("🤖 thinking..."):
                    new_advice = get_expert_advice(st.session_state.detected_result, query_to_process)
                    st.markdown(f"**You asked:** {query_to_process}")
                    st.info(new_advice)
                    new_audio = text_to_speech(new_advice)
                    st.audio(new_audio)

# ==========================================
# 6. Footer
# ==========================================
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#1b5e20; font-weight:bold;'>© 2026 Atharv Taral | Final Year Project | Savitribai Phule Pune University</p>",
    unsafe_allow_html=True
)
