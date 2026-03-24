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
    except Exception as e:
        print(f"Error fetching image: {e}")

def text_to_speech(text):
    """Convert AI advice to Audio speech"""
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    audio_path = "crop_advice.mp3"
    response.stream_to_file(audio_path)
    return audio_path

# ==========================================
# 3. Page Layout & Optimized Styling
# ==========================================
st.set_page_config(page_title="OnionGuard AI", page_icon="🧅", layout="wide")

st.markdown("""
    <style>
    /* 1. Overall App Background (Gradient to Light Gray) */
    .stApp { 
        background: linear-gradient(to right, #f8f9fa, #e9ecef) !important; 
    }
    
    /* 2. Main Title - Dark Green for high contrast */
    .main-title { 
        font-size: 60px !important;    /* Increased from 40px to 60px */
        font-weight: 900 !important;    /* Extra Bold */
        color: #1b5e20 !important;      /* Dark Green */
        text-align: center !important; 
        margin-top: -50px !important;   /* Pulls it up to reduce empty space */
        margin-bottom: 20px !important;
        letter-spacing: -1px !important; /* Makes it look like a modern brand */
        line-height: 1.2 !important;
    }
    
    /* 3. Subheaders - Professional Green */
    h3, .stMarkdown h3 { 
        color: #2e7d32 !important; 
        font-weight: 700 !important;
    }

    /* 4. AI Advice Text & Lists - Forced Dark Color */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #262730 !important; 
        font-size: 18px !important;
    }

    /* 5. Diagnosis Result Card (Pure Black with White Text) */
    .res-card { 
        background-color: #000000 !important; 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 10px solid #2e7d32; 
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 25px;
    }
    .res-card h2, .res-card p, .res-card b {
        color: #ffffff !important; 
    }

    /* 6. Sidebar Styling */
    section[data-testid="stSidebar"] { 
        background-color: #1b5e20 !important; 
    }
    section[data-testid="stSidebar"] * { 
        color: white !important; 
    }

    /* 7. Action Button Styling */
    .stButton>button { 
        background: linear-gradient(45deg, #1b5e20, #43a047) !important; 
        color: white !important; 
        border-radius: 25px !important; 
        font-weight: bold !important;
        border: none !important;
        padding: 10px 25px !important;
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
with st.container():
    st.markdown(
        """
        <div style='
            text-align: center; 
            background-color: transparent; 
            padding: 40px 0px; 
            width: 100%;
        '>
            <h1 style='
                font-size: 80px !important; 
                font-weight: 900 !important; 
                color: #1b5e20 !important; 
                margin: 0px !important;
                padding: 0px !important;
                display: block !important;
                line-height: 1 !important;
                font-family: sans-serif !important;
            '>
                🧅 OnionGuard AI
            </h1>
            <p style='
                font-size: 25px !important; 
                color: #2e7d32 !important; 
                font-weight: 600 !important;
                margin-top: 10px !important;
            '>
                Agentic Diagnostic System for Smart Agriculture
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
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

            # Displaying Result in Black Card
            st.markdown(f"""
                <div class="res-card">
                    <h2>Diagnosis Complete</h2>
                    <p><b>Detected Disease:</b> {result}</p>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            if result != 'Healthy leaves':
                with st.spinner("🤖 Consulting AI Expert..."):
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

        # --- Display Expert Results ---
        if "advice" in st.session_state and uploaded_file:
            st.markdown("---")
            st.markdown("### 📋 Treatment Expert Suggestion")

            if "med_img" in st.session_state and st.session_state.med_img:
                st.image(st.session_state.med_img,
                         caption=f"Recommended Medicine for {st.session_state.detected_result}",
                         width=300)
            else:
                st.info("ℹ️ Reference medicine photo not found.")

            st.write(st.session_state.advice)
            st.audio(st.session_state.audio)

            st.markdown("---")
            st.subheader("💬 Chat with Krishi-Mitra AI")

            user_text_query = st.chat_input("Ask a follow-up question...")
            st.write("🎤 **Or ask via voice:**")
            voice_data = mic_recorder(start_prompt="Record", stop_prompt="Stop", key="recorder")

            query_to_process = None
            if user_text_query:
                query_to_process = user_text_query
            elif voice_data:
                with open("temp_v.wav", "wb") as f:
                    f.write(voice_data['bytes'])
                transcript = client.audio.transcriptions.create(model="whisper-1", file=open("temp_v.wav", "rb"))
                query_to_process = transcript.text

            if query_to_process:
                with st.spinner("Thinking..."):
                    new_advice = get_expert_advice(st.session_state.detected_result, query_to_process)
                    st.markdown(f"**Question:** {query_to_process}")
                    st.info(new_advice)
                    st.audio(text_to_speech(new_advice))

# ==========================================
# 6. Footer
# ==========================================
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#1b5e20; font-weight:bold;'>© 2026 Atharv Taral | Final Year Project | Savitribai Phule Pune University</p>",
    unsafe_allow_html=True
)
