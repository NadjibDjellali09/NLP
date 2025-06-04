import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import librosa

# === Load models ===
model = joblib.load("final_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
svd = joblib.load("svd_transformer.pkl")

st.set_page_config(page_title="Deception Detection App", layout="wide")

# === Sidebar Navigation ===
pages = ["📊 Overview", "📁 Dataset Info", "📤 Upload & Predict"]
page = st.sidebar.radio("Navigation", pages)

# === Page: Overview ===
if page == "📊 Overview":
    st.title("Deception Detection from Voice and Transcript")
    st.write("""
        This app demonstrates a **multimodal machine learning model** that predicts whether a speaker is being **truthful or deceptive** using:
        - Audio features (pause fillers, sound features)
        - Text features (transcriptions)
        
        Built with **SVM**, **TF-IDF + SVD**, and trained on annotated trial data.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Lie_to_Me_logo.png/250px-Lie_to_Me_logo.png", width=300)

# === Page: Dataset Info ===
elif page == "📁 Dataset Info":
    st.title("Dataset Summary")

    pause_path = "/kaggle/input/deception-truthful-data/pause_filler_features.csv"
    sound_path = "/kaggle/input/deception-truthful-data/Features_From_sound.csv"

    if os.path.exists(pause_path) and os.path.exists(sound_path):
        df_pause = pd.read_csv(pause_path)
        df_sound = pd.read_csv(sound_path)

        st.subheader("Pause Filler Features")
        st.write(df_pause.head())

        st.subheader("Sound Features")
        st.write(df_sound.head())

        st.markdown("**Number of samples:**")
        st.write(f"Pause: {df_pause.shape[0]}, Sound: {df_sound.shape[0]}")
    else:
        st.warning("Dataset files not found in the expected path.")

# === Page: Upload & Predict ===
elif page == "📤 Upload & Predict":
    st.title("Upload an Audio File")
    uploaded_audio = st.file_uploader("Upload a WAV file", type=['wav'])
    transcript_text = st.text_area("Paste transcript of the speech")

    if st.button("Predict") and uploaded_audio and transcript_text:
        # === Extract audio features ===
        try:
            y, sr = librosa.load(uploaded_audio, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            rms = np.mean(librosa.feature.rms(y))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = mfccs.mean(axis=1)
            
            audio_features = [duration, zcr, rms] + list(mfccs_mean)
            audio_features = np.array(audio_features).reshape(1, -1)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            audio_features = None

        # === Extract text features ===
        try:
            tfidf_vec = vectorizer.transform([transcript_text])
            text_svd = svd.transform(tfidf_vec)
        except Exception as e:
            st.error(f"Error processing transcript: {e}")
            text_svd = None

        if audio_features is not None and text_svd is not None:
            combined_features = np.hstack((audio_features, text_svd))
            prediction = model.predict(combined_features)[0]
            st.success(f"Prediction: {prediction.upper()}")
        else:
            st.error("Could not process input properly. Please check file and text.")
