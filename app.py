import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import librosa

# Load trained components
model = joblib.load("deception_model.joblib")
vectorizer = joblib.load("vectorizer.pkl")
svd = joblib.load("svd.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# App title
st.title("🕵️‍♂️ Deception Detection App")
st.markdown("""
Welcome to the Deception Detection demo!
This app uses audio features and transcript analysis to predict if a person is being **truthful** or **deceptive**.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ['Project Overview', 'Learning Curve', 'Test a Sample'])

# Overview
if option == 'Project Overview':
    st.header("🔍 Project Overview")
    st.write("""
    **Data Used:**
    - Audio features (e.g., MFCC, pitch)
    - Pause and filler features
    - Transcripts (analyzed using TF-IDF + SVD)

    **Model:**
    - Logistic Regression with pipeline
    - Feature selection using Mutual Information
    - Trained on ~167 samples

    **Performance:**
    - Evaluated with cross-validation and test set.
    - F1 score and ROC-AUC are used as main metrics.
    """)

# Learning Curve
elif option == 'Learning Curve':
    st.header("📈 Learning Curve")
    st.image("learning_curve.png", caption="Train vs Validation F1 Scores")

# Test a Sample
elif option == 'Test a Sample':
    st.header("🧪 Test the Model")
    st.markdown("Upload a `.wav` audio file and its corresponding transcript.")

    uploaded_audio = st.file_uploader("Upload audio (.wav)", type=["wav"])
    transcript_text = st.text_area("Paste transcript text here", height=150)

    if uploaded_audio is not None and transcript_text:
        # Extract audio features (example using librosa)
        y, sr = librosa.load(uploaded_audio, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        energy = np.mean(librosa.feature.rms(y))

        # Feature vector (same order as training expected)
        audio_features = np.concatenate([mfccs, [zcr, energy]])

        # Text features
        tfidf_vec = vectorizer.transform([transcript_text])
        svd_vec = svd.transform(tfidf_vec)

        # Combine audio + text features
        full_features = np.concatenate([audio_features, svd_vec.flatten()])

        # Scale and reshape
        full_features = full_features.reshape(1, -1)

        # Make prediction
        pred_label = model.predict(full_features)[0]
        pred_proba = model.predict_proba(full_features)[0]

        label = label_encoder.inverse_transform([pred_label])[0]
        st.success(f"**Prediction:** {label.upper()} ({pred_proba[pred_label]*100:.1f}% confidence)")

