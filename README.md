# 🧅 OnionGuard: Agentic AI-Based Crop Diagnostic System

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![Model Accuracy](https://img.shields.io/badge/accuracy-96.09%25-brightgreen.svg)]()

**OnionGuard AI** is an advanced 'Agentic AI' project designed to detect and diagnose 10+ different onion leaf diseases with high precision. It doesn't just classify diseases; it acts as an autonomous **Agriculture Expert**, providing detailed treatment advice and interacting with farmers via voice.

---

## 🚀 Key Features

* **Deep Learning Diagnosis:** Uses a CNN (Convolutional Neural Network) model to scan onion leaves with **96.09%** accuracy.
* **Agentic Expert Advice:** Powered by **GPT-4o**, the system generates expert treatment plans, including disease causes, recommended fungicides, and application methods.
* **Voice-Enabled Interaction:** Integrated with **Whisper API** to understand farmers' queries via microphone and respond using **Text-to-Speech (TTS)**.
* **Multimodal Input:** Supports Text (Chat), Voice (Speech), and Image (Scanning) inputs for a seamless user experience.
* **Medicine Reference Engine:** Built-in capability to fetch reference images of recommended medicines using **Google Custom Search API**.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Web-based Interactive Dashboard)
- **Deep Learning:** TensorFlow, Keras (CNN Model)
- **Generative AI:** OpenAI GPT-4o (Autonomous Reasoning Agent)
- **Speech AI:** OpenAI Whisper (Speech-to-Text) & OpenAI TTS (Audio Output)
- **Programming:** Python 3.13
- **Data Handling:** NumPy, Pandas, Pillow (PIL), Python-dotenv

---

## 📂 Project Structure

```text
Onion/
├── .env                # Private API Keys (OpenAI, Google) - [Hidden]
├── best_model.h5       # Pre-trained CNN Model File
├── onion.py            # Main Application Logic (Streamlit)
├── requirements.txt    # List of Project Dependencies
├── .gitignore          # Files excluded from GitHub for security
└── README.md           # Project Documentation (This file)
