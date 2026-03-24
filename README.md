# 🧅 OnionGuard: Agentic AI-Based Crop Diagnostic System

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![Model Accuracy](https://img.shields.io/badge/accuracy-96.09%25-brightgreen.svg)]()

## 🌐 Live Demo
👉 [Click here to view the Live App](https://onionguard-ai-diagnostics-1232.streamlit.app/)

**OnionGuard AI** is a state-of-the-art 'Agentic AI' solution for modern agriculture. It identifies 11 types of onion diseases using computer vision and acts as an autonomous **Agri-Expert** that provides localized treatment advice and voice-based interaction for farmers.

---

## 🚀 Key Features

* **Deep Learning Diagnosis:** CNN-based image classification with **96.09% accuracy**.
* **Agentic Reasoning:** Powered by **GPT-4o**, the system provides expert-level treatment logic.
* **Voice-First Interface:** Uses **OpenAI Whisper** for speech-to-text and **TTS** for audio responses.
* **Medicine Reference Engine:** Automatically attempts to fetch visual references of recommended fungicides via Google API.

---

## 🛠️ Tech Stack

-   **Frontend:** Streamlit
-   **Core AI:** TensorFlow/Keras (CNN)
-   **Agentic Brain:** OpenAI GPT-4o
-   **Speech Engine:** Whisper-1 & TTS-1
-   **Language:** Python 3.10
-   **Environment:** Python-dotenv, Requests, Pillow

---

## ⚙️ How It Works (The Agentic Workflow)

The project follows a multi-stage **Agentic Workflow** to assist farmers:

1.  **Image Perception:** The user uploads a leaf scan. The CNN model classifies the disease from 11 pre-defined classes.
2.  **Autonomous Reasoning:** Once a disease is detected, the **GPT-4o Agent** analyzes the result and generates a professional treatment plan (Causes, Medicines, Application).
3.  **Tool Execution:** The system triggers a **Google Custom Search Tool** to fetch real-world medicine bottle images for the farmer's reference.
4.  **Voice Feedback:** The advice is converted into an audio file so farmers can listen to the instructions without reading.
5.  **Interactive Chat:** A chatbot allows the farmer to ask follow-up questions about the treatment via text or voice.

---

## 📂 Project Structure

```text
Onion/
├── .devcontainer/      # Codespaces/Container configuration
├── .gitattributes      # Git LFS/Attribute settings
├── README.md           # Project Documentation (English)
├── best_model.h5       # Trained CNN Model (96.09% Accuracy)
├── class_labels.txt    # List of 11 Onion Disease classes
├── onion.py            # Main Agentic AI Application (Streamlit)
├── requirements.txt    # Project Dependencies (Optimized for Python 3.10)
└── update_profile.jpg  # Profile/Developer Image for UI
