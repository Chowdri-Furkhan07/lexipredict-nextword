"""
app.py
------
LexiPredict — Next Word Prediction Studio
A Streamlit interface for comparing RNN / LSTM / BiLSTM next-word predictors,
generating multi-word continuations, and inspecting model confidence.
"""

import os
import pickle

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from src import config
from src.predict import generate_text, top_k_predictions

# ---------------------------------------------------------------------------
# PAGE CONFIG + DARK THEME STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LexiPredict | Next Word Prediction Studio",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    .lp-card {
        background-color: #161b22;
        border: 1px solid #262c36;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
    }
    .lp-title { font-size: 2rem; font-weight: 700; color: #f5f5f5; }
    .lp-subtitle { color: #9aa4b2; font-size: 0.95rem; margin-bottom: 1.2rem; }
    .lp-badge {
        display: inline-block; background-color: #1f6feb22; color: #58a6ff;
        border: 1px solid #1f6feb55; border-radius: 999px;
        padding: 0.15rem 0.7rem; font-size: 0.78rem; margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models = {}
    paths = {
        "SimpleRNN": config.RNN_MODEL_PATH,
        "LSTM": config.LSTM_MODEL_PATH,
        "BiLSTM": config.BILSTM_MODEL_PATH,
    }
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = load_model(path)
    return models


@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(config.TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(config.MAX_LEN_PATH, "rb") as f:
        max_len = pickle.load(f)
    return tokenizer, max_len


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown('<div class="lp-title">🧠 LexiPredict</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lp-subtitle">Next-word prediction studio powered by SimpleRNN, LSTM, and BiLSTM'
    ' models trained on a custom corpus.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="lp-badge">TensorFlow / Keras</span>'
    '<span class="lp-badge">Streamlit</span>'
    '<span class="lp-badge">NLP</span>'
    '<span class="lp-badge">Sequence Modeling</span>',
    unsafe_allow_html=True,
)
st.write("")

if not os.path.exists(config.TOKENIZER_PATH):
    st.error(
        "No trained artifacts found. Run `python -m src.train` first to "
        "generate the models and tokenizer."
    )
    st.stop()

models = load_models()
tokenizer, max_len = load_artifacts()

if not models:
    st.error("No model files found in the `models/` directory.")
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Model", list(models.keys()))
    compare_all = st.checkbox("Compare all models side by side", value=False)
    n_words = st.slider("Words to generate", 1, 15, config.DEFAULT_WORDS_TO_GENERATE)
    temperature = st.slider(
        "Temperature (creativity)", 0.0, 1.5, config.DEFAULT_TEMPERATURE, 0.1,
        help="0 = deterministic / most likely word. Higher = more diverse predictions.",
    )
    st.caption(
        "Trained on a small demo corpus (see `data/corpus.txt`). "
        "Swap in a larger dataset for production-quality predictions."
    )

# ---------------------------------------------------------------------------
# MAIN INPUT
# ---------------------------------------------------------------------------
input_text = st.text_input("Enter a sentence prompt", placeholder="e.g. deep learning is")

col_predict, col_generate = st.columns(2)
predict_clicked = col_predict.button("🔮 Predict Next Word", use_container_width=True)
generate_clicked = col_generate.button("✍️ Generate Continuation", use_container_width=True)

if (predict_clicked or generate_clicked) and not input_text.strip():
    st.warning("Please enter a sentence first.")

# ---------------------------------------------------------------------------
# SINGLE-MODEL VIEW
# ---------------------------------------------------------------------------
if input_text.strip() and not compare_all:
    model = models[model_choice]

    if predict_clicked:
        st.markdown('<div class="lp-card">', unsafe_allow_html=True)
        st.subheader(f"Top predictions — {model_choice}")
        results = top_k_predictions(model, tokenizer, max_len, input_text, k=5)
        df = pd.DataFrame(results, columns=["word", "confidence"])
        st.bar_chart(df.set_index("word"))
        st.success(f"Most likely next word: **{results[0][0]}**")
        st.markdown("</div>", unsafe_allow_html=True)

    if generate_clicked:
        st.markdown('<div class="lp-card">', unsafe_allow_html=True)
        st.subheader(f"Generated continuation — {model_choice}")
        generated = generate_text(model, tokenizer, max_len, input_text, n_words, temperature)
        st.write(f"**{generated}**")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# COMPARISON VIEW
# ---------------------------------------------------------------------------
if input_text.strip() and compare_all:
    cols = st.columns(len(models))
    for col, (name, model) in zip(cols, models.items()):
        with col:
            st.markdown('<div class="lp-card">', unsafe_allow_html=True)
            st.markdown(f"**{name}**")
            if predict_clicked:
                results = top_k_predictions(model, tokenizer, max_len, input_text, k=3)
                for word, score in results:
                    st.write(f"`{word}` — {score:.1%}")
            if generate_clicked:
                generated = generate_text(model, tokenizer, max_len, input_text, n_words, temperature)
                st.write(generated)
            st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption(
    "LexiPredict · Built with TensorFlow/Keras + Streamlit · "
    "See README.md for architecture details and training instructions."
)
