"""
predict.py
----------
Inference utilities shared by the Streamlit app and any CLI usage.
Adds temperature-based sampling and multi-word generation on top of the
original single-word, always-argmax prediction.
"""

import pickle

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src import config


def load_tokenizer_and_max_len():
    with open(config.TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(config.MAX_LEN_PATH, "rb") as f:
        max_len = pickle.load(f)
    return tokenizer, max_len


def _sample_with_temperature(probabilities: np.ndarray, temperature: float) -> int:
    """
    Lower temperature -> more confident/greedy choices.
    Higher temperature -> more diverse/creative choices.
    temperature <= 0 falls back to plain argmax.
    """
    if temperature <= 0:
        return int(np.argmax(probabilities))

    probabilities = np.asarray(probabilities).astype("float64")
    log_probs = np.log(probabilities + 1e-9) / temperature
    exp_probs = np.exp(log_probs)
    scaled = exp_probs / np.sum(exp_probs)
    return int(np.random.choice(len(scaled), p=scaled))


def predict_next_word_distribution(model, tokenizer, max_len, text: str):
    """Returns the raw probability distribution over the vocabulary."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
    probabilities = model.predict(token_list, verbose=0)[0]
    return probabilities


def top_k_predictions(model, tokenizer, max_len, text: str, k: int = 5):
    """Return the top-k candidate next words with their confidence scores."""
    probabilities = predict_next_word_distribution(model, tokenizer, max_len, text)
    top_indices = np.argsort(probabilities)[-k:][::-1]

    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    results = [
        (index_to_word.get(i, "<unknown>"), float(probabilities[i]))
        for i in top_indices
    ]
    return results


def predict_next_word(model, tokenizer, max_len, text: str, temperature: float = 0.0, avoid_words=None) -> str:
    probabilities = predict_next_word_distribution(model, tokenizer, max_len, text)

    if avoid_words:
        probabilities = probabilities.copy()
        for word in avoid_words:
            avoid_index = tokenizer.word_index.get(word)
            if avoid_index is not None and avoid_index < len(probabilities):
                probabilities[avoid_index] = 0.0

    predicted_index = _sample_with_temperature(probabilities, temperature)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""


def generate_text(model, tokenizer, max_len, seed_text: str, n_words: int = 5, temperature: float = 0.0, repeat_window: int = 3) -> str:
    """
    Iteratively predicts and appends n_words to the seed text.
    Blocks recently-used words (default: last 3) from being picked again —
    a lightweight repetition penalty that prevents small greedy-decoded
    models from looping (e.g. "the a the a...").
    """
    text = seed_text
    generated_words = seed_text.strip().lower().split()

    for _ in range(n_words):
        recent = generated_words[-repeat_window:] if repeat_window > 0 else []
        next_word = predict_next_word(model, tokenizer, max_len, text, temperature, avoid_words=recent)
        if not next_word:
            break
        text += " " + next_word
        generated_words.append(next_word)
    return text
