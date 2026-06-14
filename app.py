import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set up a wide, dark-themed page profile
st.set_page_config(page_title="Sentify AI", page_icon="🎬", layout="centered")

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

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('sentiment_gru_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except:
        return None, None

model, tokenizer = load_artifacts()

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# --- APP HEADER ---
st.markdown("<h1>Sentify.AI</h1>", unsafe_allow_html=True)
# Replaced the old tagline with your new custom one right here
st.markdown("<p style='color: #a5b4fc; font-size:16px; margin-top: -10px; font-weight: 500;'>✨ Know the emotion of reviews</p>", unsafe_allow_html=True)
st.write("") # Clean, controlled spacing

# --- INTERACTIVE CONTENT BLOCK ---
# Using a native container with a border eliminates the empty ghost boxes completely!
with st.container(border=True):
    st.markdown("<p style='font-weight: 500; margin-bottom: 5px;'>💡 Quick Test Templates:</p>", unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns(3)

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    if col_c1.button("✨ Masterpiece!"):
        st.session_state.input_text = "Absolute masterpiece! The cinematography was stellar and the acting was top notch."
    if col_c2.button("🗑️ Garbage & Never"):
        st.session_state.input_text = "garbage it is and i am never gonna watch movie from now on."
    if col_c3.button("🔄 Never Regret"):
        st.session_state.input_text = "i am never gonna regret watching this movie."

    # Main input field
    user_review = st.text_area("Write or paste your movie review below:", value=st.session_state.input_text, height=120)

# --- INFERENCE EXECUTION ---
st.write("")
if st.button("🚀 Compute ", type="primary", use_container_width=True):
    if not user_review.strip():
        st.warning("Please input some text context first.")
    elif model is None or tokenizer is None:  # Bone fix verified here!
        st.error("Error: Model weights or tokenizer pickle missing from directory folder.")
    else:
        cleaned_input = clean_text(user_review)
        sequence = tokenizer.texts_to_sequences([cleaned_input])
        padded_sequence = pad_sequences(sequence, maxlen=300, padding='post', truncating='pre')
        
        with st.spinner("Evaluating contextual hidden states..."):
            prediction = model.predict(padded_sequence)[0][0]
        
        # Results Presentation Card
        with st.container(border=True):
            st.subheader("📊 Analytical Metrics")
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