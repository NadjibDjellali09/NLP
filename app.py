import streamlit as st
import joblib
import numpy as np
import pandas as pd
import librosa
import os
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for clean UI
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Deception Detection App", layout="centered")
st.title("🕵️‍♂️ Deception Detection App")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📄 Project Overview", "📈 Learning Curve", "🧪 Test the Model"])

# Try loading models
try:
    model = joblib.load("deception_model.joblib")
    vectorizer = joblib.load("vectorizer.pkl")
    svd = joblib.load("svd.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    model_loaded = True
except FileNotFoundError as e:
    st.error("🚫 One or more required model files are missing.")
    st.info("Please make sure these files are in the same folder as this app:\n"
            "`deception_model.joblib`, `vectorizer.pkl`, `svd.pkl`, `label_encoder.pkl`")
    model_loaded = False

# 1. Project Overview Page
if page == "📄 Project Overview":
    st.header("🔍 Project Summary")
    st.markdown("""
    This app demonstrates a **multimodal deception detection system** using:
    
    - **Audio Features:** MFCCs, energy, zero-crossing rate, etc.
    - **Text Features:** TF-IDF of transcripts reduced using SVD
    - **Classifier:** Logistic Regression with feature selection
    
    **Training Highlights:**
    - ~167 samples
    - Cross-validated F1 score and ROC-AUC
    - Trained using `SelectKBest` for top 10 features
    
    You can test the model by uploading a `.wav` file and entering the transcript.
    """)

# 2. Learning Curve Page
elif page == "📈 Learning Curve":
    st.header("📈 Learning Curve")
    if os.path.exists("learning_curve.png"):
        st.image("learning_curve.png", caption="F1 Score vs Training Size", use_column_width=True)
    else:
        st.warning("Learning curve plot not found.")

# 3. Test the Model Page
elif page == "🧪 Test the Model":
    st.header("🎤 Upload Audio and Transcript")
    st.markdown("Upload a `.wav` file and paste its transcript. The model will predict whether the speech is **truthful** or **deceptive**.")

    if not model_loaded:
        st.warning("Model not loaded. Upload required files to use this section.")
    else:
        uploaded_audio = st.file_uploader("Upload .wav audio file", type=["wav"])
        transcript_text = st.text_area("Paste transcript text here", height=150)

        if uploaded_audio is not None and transcript_text.strip() != "":
            try:
                # Load audio
                y, sr = librosa.load(uploaded_audio, sr=None)

                # Extract audio features (basic MFCC + energy + ZCR)
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                energy = np.mean(librosa.feature.rms(y))

                # Audio feature vector
                audio_features = np.concatenate([mfccs, [zcr, energy]])

                # Text features
                tfidf_vec = vectorizer.transform([transcript_text])
                svd_vec = svd.transform(tfidf_vec)

                # Combine
                full_input = np.concatenate([audio_features, svd_vec.flatten()])
                full_input = full_input.reshape(1, -1)

                # Predict
                pred_label = model.predict(full_input)[0]
                pred_proba = model.predict_proba(full_input)[0]

                label = label_encoder.inverse_transform([pred_label])[0]
                confidence = pred_proba[pred_label] * 100

                st.success(f"🔍 **Prediction:** `{label.upper()}`")
                st.metric("Confidence", f"{confidence:.1f}%")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.info("Please upload a valid audio file and paste a transcript.")
