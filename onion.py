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

# Loading environment variables from the .env file for security
load_dotenv()

# Retrieving the OpenAI API Key from the local environment
API_KEY = os.getenv("OPENAI_API_KEY")

# Safety Check: Terminate execution if the API Key is missing or invalid
if not API_KEY:
    # Displaying a clear error message on the Streamlit UI
    st.error("❌ OpenAI API Key not found in .env file!")
    # Halting the application to prevent further logic errors
    st.stop()

# Set to True to enable GPT-4o expert advice; set to False for Demo Mode (saves API credits)
API_ENABLED = False


# Initializing the OpenAI Client with the authenticated API Key to enable AI Agent features
client = openai.OpenAI(api_key=API_KEY)

# ==========================================
# 2. Expert AI Advisor Functions
# ==========================================
def get_expert_advice(disease_name, user_query="What is the treatment?"):
    """
    Fetch medicine and treatment advice from GPT-4o
    """
    
    # Constructing a detailed prompt for the AI Agent to act as an Agri-Scientist
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
    
    # Sending the request to OpenAI API with specific system and user roles
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful Agri-Expert."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extracting and returning the text content from the AI response
    return response.choices[0].message.content


import requests
def get_medicine_image(query):
    """Searching for a medicine photo from Google"""
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CX")
    search_url = "https://www.googleapis.com/customsearch/v1"

    # Define search parameters: focusing on agricultural medicine/bottles
    params = {
        'q': query + " agriculture medicine fungicide bottle",
        'cx': cx,
        'key': api_key,
        'searchType': 'image', # Tell Google we only want image results
        'num': 1               # Fetch only the top most relevant image
    }
    try:
        # Execute the GET request to Google API
        response = requests.get(search_url, params=params)
        data = response.json()
        # Extract the image link from the JSON response items
        if 'items' in data:
            return data['items'][0]['link']
    except Exception as e:
        # Log any errors during the API call or data processing
        print(f"Error fetching image: {e}")

def text_to_speech(text):
    """Convert AI advice to Audio speech"""
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    audio_path = "crop_advice.mp3"
    response.stream_to_file(audio_path)
    return audio_path

# ==========================================
# 3. GLOBAL PREMIUM STYLING (Forced Light Mode)
# ==========================================
st.markdown("""
    <style>
    /* 1. Global App Background: Forcing a clean, professional light-gray theme */
    .stApp {
        background-color: #f8f9fa !important;
    }

    /* 2. Subheaders: Ensuring high contrast and bold appearance for readability */
    h3, .stMarkdown h3 { 
        color: #1a1a1a !important; 
        font-weight: 800 !important;
        text-shadow: none !important;
    }

    /* 3. Notifications: Improving text visibility inside warning/info boxes */
    [data-testid="stNotification"] p {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* 4. Sidebar: Customizing with Deep Forest Green to match Agriculture theme */
    /* Sidebar - Deep Forest Green */
    section[data-testid="stSidebar"] {
        background-color: #1b5e20 !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* 5. Header Bar: Blending the top bar with the main app background */
    header[data-testid="stHeader"] {
        background-color: #f8f9fa !important;
    }

    /* 6. Text Visibility: Standardizing font size and weight for all markdown elements */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #1a1a1a !important;
        font-size: 18px !important;
    }

    /* 7. Inputs & Labels: Darkening labels for better accessibility */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, label {
        color: #121212 !important; 
        font-weight: 500 !important;
    }

    .stFileUploader label {
        color: #1a1a1a !important;
    }

    /* 8. Hero Section: Centering the brand title and managing top whitespace */
    .hero-section {
        text-align: center;
        padding: 60px 0 20px 0;
        margin-top: -100px;
        width: 100%;
    }

    /* 9. Main Title: Large-scale typography for high-impact branding */
    .hero-title {
        font-size: 71px !important;
        font-weight: 900 !important;
        color: #1b5e20 !important;
        line-height: 1 !important;
        letter-spacing: -3px !important;
        margin-bottom: 10px !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    /* 10. Hero Subtitle: Secondary branding line with a lighter green contrast */
    .hero-subtitle {
        font-size: 24px !important;
        color: #43a047 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

 

# ==========================================
# 4. Sidebar & Model Loading
# ==========================================
# Configuring the Sidebar for User Information and Project Metrics
with st.sidebar:
    # Displaying the student profile image and project metadata
    st.image("update_profile.jpg", width=200)
    st.title("Project Dashboard")
    
    # Displaying the performance metric of the trained CNN model
    st.metric(label="Model Accuracy", value="96.09%")
    st.write("**Student:** Atharv Taral")
    st.write("**Domain:** Smart Agriculture AI")

# Utilizing @st.cache_resource to ensure the model loads only once (Optimization)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("best_model.h5")

# Error Handling block to catch issues during model initialization
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Mapping of integer indices to human-readable class names for the 11 onion diseases
class_labels = [
    'Alternaria_D', 'Botrytis Leaf Blight', 'Bulb_blight-D', 'Caterpillar-P',
    'Fusarium-D', 'Healthy leaves', 'Iris yellow virus_augment', 'Rust',
    'Virosis-D', 'Xanthomonas Leaf Blight', 'stemphylium Leaf Blight'
]

# ==========================================
# 5. Main Dashboard Header
# ==========================================
# Injecting Custom HTML for Professional Branding and Hero Section
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">🧅 OnionGuard AI</div>
        <div class="hero-subtitle">Agentic Diagnostic System for Smart Agriculture</div>
    </div>
    """, 
    unsafe_allow_html=True
)
# Adding a visual separator (Horizontal Rule) for UI clean-up
st.markdown("---")

# Creating a two-column layout for a balanced and professional Dashboard UI
col1, col2 = st.columns([1, 1], gap="large")

# --- COLUMN 1: IMAGE UPLOAD SECTION ---
with col1:
    st.subheader("📤 Upload Leaf Scan")
    # File uploader restricted to common image formats
    uploaded_file = st.file_uploader("Select an image of the onion leaf", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "heic"]
                                     
    if uploaded_file:
        # Displaying the uploaded image as a preview for the user
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Preview', use_container_width=True)

# --- COLUMN 2: ANALYSIS & AI DIAGNOSIS SECTION ---
with col2:
    st.subheader("🔍 Analysis & Expert Advice")
    if uploaded_file:
        # Pre-processing the image to match the CNN model's input requirements (100x100)
        img_resized = img.resize((100, 100))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Triggering the AI Diagnosis upon button click
        if st.button('✨ Start AI Diagnosis'):
            # Visual feedback with a progress bar for simulated processing time
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent + 1)

            # Executing Model Prediction and calculating confidence score
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions)
            result = class_labels[class_idx]
            confidence = np.max(predictions) * 100

            # 1. Displaying the Result Card using custom CSS for impactड
            st.markdown(f"""
                <div class="res-card">
                    <h2>Diagnosis Complete</h2>
                    <p><b>Detected Disease:</b> {result}</p>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

        # 2. Safety Lock & Agentic AI Logicा
        if result != 'Healthy leaves':
            # Checking the API_ENABLED toggle to manage OpenAI credits
            if not API_ENABLED:
                # Custom Alert Box for Demo Mode (Safe & Informative)
                st.markdown("""
                    <div style="
                        background-color: #fff3cd; 
                        color: #856404; 
                        padding: 18px; 
                        border-radius: 12px; 
                        border: 2px solid #ffeeba;
                        font-size: 18px;
                        font-weight: bold;
                        display: flex;
                        align-items: center;
                        margin-bottom: 20px;
                    ">
                        <span style="font-size: 24px; margin-right: 15px;">⚠️</span>
                        Demo Mode: AI Expert Advice is currently disabled to save API credits. Prediction only.
                    </div>
                """, unsafe_allow_html=True)
            else:
               # Proceeding with Agentic AI services (GPT-4o Advice, Medicine Search, & TTS)
                with st.spinner("🤖 Consulting AI Expert..."):
                    advice = get_expert_advice(result)
                    st.session_state.advice = advice
                    st.session_state.detected_result = result
                    
                    # Fetching visual and audio metadata
                    med_img = get_medicine_image(result)
                    st.session_state.med_img = med_img
                    
                    audio_path = text_to_speech(advice)
                    st.session_state.audio = audio_path
        else:
            # Celebrating healthy results with Streamlit balloons
            st.balloons()
            st.success("The crop appears to be in excellent health!")

         # --- AGENTIC OUTPUTS & INTERACTIVE CHAT SECTION ---
         # Persisting the advice results across app re-runs using session state
        if "advice" in st.session_state and uploaded_file:
            st.markdown("---")
            st.markdown("### 📋 Treatment Expert Suggestion")

            # Displaying the dynamically fetched medicine image
            if "med_img" in st.session_state and st.session_state.med_img:
                st.image(st.session_state.med_img,
                         caption=f"Recommended Medicine for {st.session_state.detected_result}",
                         width=300)
            else:
                st.info("ℹ️ Reference medicine photo not found.")

            # Rendering text advice and generating audio playback
            st.write(st.session_state.advice)
            st.audio(st.session_state.audio)

            # --- MULTI-MODAL CHATBOT (KRISHI-MITRA) ---
            st.markdown("---")
            st.subheader("💬 Chat with Krishi-Mitra AI")

            # Support for both Text input and Voice recording (Whisper-1 Speech-to-Text)
            user_text_query = st.chat_input("Ask a follow-up question...")
            st.write("🎤 **Or ask via voice:**")
            voice_data = mic_recorder(start_prompt="Record", stop_prompt="Stop", key="recorder")

            query_to_process = None
            if user_text_query:
                query_to_process = user_text_query
            elif voice_data:
                # Handling audio buffer and transcribing using Whisper API
                with open("temp_v.wav", "wb") as f:
                    f.write(voice_data['bytes'])
                transcript = client.audio.transcriptions.create(model="whisper-1", file=open("temp_v.wav", "rb"))
                query_to_process = transcript.text

            # Processing the final query and displaying AI follow-up response
            if query_to_process:
                with st.spinner("Thinking..."):
                    new_advice = get_expert_advice(st.session_state.detected_result, query_to_process)
                    st.markdown(f"**Question:** {query_to_process}")
                    st.info(new_advice)
                    st.audio(text_to_speech(new_advice))

# ==========================================
# 6. Footer
# ==========================================
# Adding visual separation at the bottom of the page
st.markdown("<br><br><hr>", unsafe_allow_html=True)

# Injecting Custom HTML for the Copyright and Academic Branding
st.markdown(
    "<p style='text-align:center; color:#1b5e20; font-weight:bold;'>© 2026 Atharv Taral | Final Year Project | Savitribai Phule Pune University</p>",
    unsafe_allow_html=True # Essential for rendering the center-aligned styled paragraph
)
