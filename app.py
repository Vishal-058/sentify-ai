import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set up a wide, dark-themed page layout
st.set_page_config(page_title="Sentify AI", layout="centered")

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    /* Main card container styling applied to native containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    }
    /* Styled Headers */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    /* Metric Card Custom Design */
    .sentiment-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    .pos-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        color: #10b981;
    }
    .neg-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        color: #ef4444;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG ---
# NOTE: This must match the maxlen used during training (cell 2: max_length = 300)
MAX_LENGTH = 300

# NOTE: Make sure this filename matches whatever your training notebook saved
# (e.g. 'sentiment_gru_model.h5' or 'sentiment_gru_model.keras')
MODEL_PATH = "sentiment_gru_model.keras"
TOKENIZER_PATH = "tokenizer.pickle"


# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.warning(f"Could not load model/tokenizer: {e}")
        return None, None


model, tokenizer = load_artifacts()


# --- TEXT CLEANING ---
# IMPORTANT: This must exactly match the clean_text() function used during
# training (notebook cell 2). If you change this, you must also re-run
# preprocessing/tokenizer-fitting/training so the model sees consistent input.
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)      # Strip HTML tags like <br />
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # Strip punctuation and numbers
    text = text.lower().strip()               # Lowercase + trim
    return text


# --- APP HEADER ---
st.markdown("<h1>Sentify.AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #a5b4fc; font-size:16px; margin-top: -10px; font-weight: 500;'>✨ Know the emotion of reviews</p>", unsafe_allow_html=True)
st.write("")

# --- INTERACTIVE CONTENT BLOCK ---
with st.container(border=True):
    st.markdown("<p style='font-weight: 500; margin-bottom: 5px;'>💡 Quick Test Templates:</p>", unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns(3)

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    if col_c1.button("Masterpiece!"):
        st.session_state.input_text = "Absolute masterpiece! The cinematography was stellar and the acting was top notch."
    if col_c2.button("Garbage"):
        st.session_state.input_text = "garbage it is."
    if col_c3.button("Best"):
        st.session_state.input_text = "it is the best movie that i ever watch in my over life."

    # Main input field
    user_review = st.text_area("Write or paste your movie review below:", value=st.session_state.input_text, height=120)

# --- INFERENCE EXECUTION ---
st.write("")
if st.button(" Compute ", type="primary", use_container_width=True):
    if not user_review.strip():
        st.warning("Please input some text context first.")
    elif model is None or tokenizer is None:
        st.error("Error: Model weights or tokenizer pickle missing from directory folder.")
    else:
        # Preprocess text and generate initial sequences
        cleaned_input = clean_text(user_review)
        sequence = tokenizer.texts_to_sequences([cleaned_input])

        # 🛑 GIBBERISH / OOV GUARDRAIL
        # Intercepts strings that break into nothing or map entirely to the OOV token index (1)
        if len(sequence[0]) == 0 or all(token == 1 for token in sequence[0]):
            st.error("🤔 **Low Confidence Detection:** The system cannot recognize these words. Please enter a valid sentence!")
        else:
            # ⚠️ SHORT INPUT WARNING
            # This model was trained on full-length movie reviews (~230 words avg).
            # Very short inputs (especially short negations like "not good") fall
            # outside that distribution and predictions are less reliable.
            if len(cleaned_input.split()) < 5:
                st.info("⚠️ **Heads up:** Short inputs (especially negations like 'not good') are less reliable. Try writing a fuller sentence for a more accurate result.")
            # Padding must match training exactly: padding='post', truncating='pre'
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='pre')

            with st.spinner("Evaluating contextual hidden states..."):
                prediction = model.predict(padded_sequence)[0][0]

            # Results Presentation Card
            with st.container(border=True):
                st.subheader(" Analytical Metrics")
                col1, col2 = st.columns([1, 1.2], gap="large")

                if prediction >= 0.5:
                    confidence = prediction * 100
                    st.balloons()
                    with col1:
                        st.markdown(f"<div class='sentiment-box pos-box'>🟢 POSITIVE SENTIMENT</div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<p style='margin-bottom:2px; font-size:14px; color:#94a3b8;'>Model Confidence Rating</p>", unsafe_allow_html=True)
                        st.progress(int(confidence))
                        st.markdown(f"<h3 style='margin-top:0px; color:#10b981;'>{confidence:.2f}%</h3>", unsafe_allow_html=True)
                else:
                    confidence = (1 - prediction) * 100
                    with col1:
                        st.markdown(f"<div class='sentiment-box neg-box'>🔴 NEGATIVE SENTIMENT</div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<p style='margin-bottom:2px; font-size:14px; color:#94a3b8;'>Model Confidence Rating</p>", unsafe_allow_html=True)
                        st.progress(int(confidence))
                        st.markdown(f"<h3 style='margin-top:0px; color:#ef4444;'>{confidence:.2f}%</h3>", unsafe_allow_html=True)