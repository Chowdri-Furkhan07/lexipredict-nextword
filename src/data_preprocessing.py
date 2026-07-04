"""
data_preprocessing.py
----------------------
Loads the raw text corpus, tokenizes it, builds n-gram training sequences,
and produces the train/validation split used by train.py.

Works with real prose (paragraphs of running text), not just one-sentence-
per-line files: raw text is flattened and split into sentences using
punctuation, then each sentence becomes a family of n-gram sequences.
"""

import os
import pickle
import re

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src import config

# Sentences shorter than this carry too little context to be useful;
# sentences longer than this blow up max_len (and therefore padding) for
# every other example, so we cap both ends.
MIN_SENTENCE_WORDS = 4
MAX_SENTENCE_WORDS = 25


def load_corpus(path: str = config.CORPUS_PATH) -> str:
    """Read the raw corpus from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus not found at {path}. Add a text file (plain prose or "
            f"one sentence per line), or point config.CORPUS_PATH to your "
            f"own dataset."
        )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def split_into_sentences(text: str):
    """
    Flatten paragraphs/newlines and split raw prose into sentences on
    '.', '!', '?'. This lets the pipeline train on real books/articles,
    not just pre-formatted one-sentence-per-line files.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    # Split on sentence-ending punctuation, keeping it simple/robust
    # rather than handling every edge case (abbreviations, etc.) — the
    # occasional partial sentence has minimal impact on training signal.
    raw_sentences = re.split(r"(?<=[.!?])\s+", flat)

    sentences = []
    for s in raw_sentences:
        s = s.strip(" \"'“”‘’")
        if not s:
            continue
        word_count = len(s.split())
        if MIN_SENTENCE_WORDS <= word_count <= MAX_SENTENCE_WORDS:
            sentences.append(s)
    return sentences


def build_tokenizer(sentences) -> Tokenizer:
    """Fit a Keras tokenizer on the corpus sentences."""
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def build_sequences(sentences, tokenizer: Tokenizer):
    """
    Turn each sentence into progressively growing n-gram sequences.
    e.g. "deep learning is fun" ->
        [deep, learning] -> [deep, learning, is] -> [deep, learning, is, fun]
    This is what teaches the model to predict the *next* word at every step.
    """
    input_sequences = []

    for sentence in sentences:
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(token_list)):
            input_sequences.append(token_list[: i + 1])

    if not input_sequences:
        raise ValueError("No valid sequences could be built from the corpus.")

    max_len = max(len(seq) for seq in input_sequences)
    padded = np.array(
        pad_sequences(input_sequences, maxlen=max_len, padding="pre")
    )

    X, y = padded[:, :-1], padded[:, -1]
    return X, y, max_len


def prepare_data():
    """
    Full pipeline: load corpus -> split into sentences -> tokenize ->
    build sequences -> train/val split. Also persists the tokenizer and
    max_len so the app can reuse them at inference time.
    """
    text = load_corpus()
    sentences = split_into_sentences(text)
    if not sentences:
        raise ValueError(
            "No sentences survived length filtering — check the corpus "
            "content or relax MIN_SENTENCE_WORDS / MAX_SENTENCE_WORDS."
        )

    tokenizer = build_tokenizer(sentences)
    total_words = len(tokenizer.word_index) + 1

    X, y, max_len = build_sequences(sentences, tokenizer)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED
    )

    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    with open(config.TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(config.MAX_LEN_PATH, "wb") as f:
        pickle.dump(max_len, f)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "total_words": total_words,
        "max_len": max_len,
        "tokenizer": tokenizer,
        "num_sentences": len(sentences),
    }


if __name__ == "__main__":
    data = prepare_data()
    print(f"Sentences used  : {data['num_sentences']}")
    print(f"Vocabulary size : {data['total_words']}")
    print(f"Max sequence len: {data['max_len']}")
    print(f"Train samples   : {len(data['X_train'])}")
    print(f"Val samples     : {len(data['X_val'])}")
