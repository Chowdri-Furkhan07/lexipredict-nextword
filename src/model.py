"""
model.py
--------
Model architectures for next-word prediction: a SimpleRNN baseline,
an LSTM model, and a Bidirectional LSTM model. Each includes Dropout
and an Adam optimizer with a configurable learning rate, which the
original tutorial-style script lacked.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Input,
    LSTM,
    SimpleRNN,
    Bidirectional,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

from src import config


def build_rnn_model(total_words: int, max_len: int) -> Sequential:
    model = Sequential(name="SimpleRNN_NextWord")
    model.add(Input(shape=(max_len - 1,)))
    model.add(Embedding(total_words, config.EMBEDDING_DIM))
    model.add(SimpleRNN(config.RNN_UNITS))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(Dense(total_words, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def build_lstm_model(total_words: int, max_len: int) -> Sequential:
    model = Sequential(name="LSTM_NextWord")
    model.add(Input(shape=(max_len - 1,)))
    model.add(Embedding(total_words, config.EMBEDDING_DIM))
    model.add(LSTM(config.LSTM_UNITS, return_sequences=False))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(Dense(total_words, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def build_bilstm_model(total_words: int, max_len: int) -> Sequential:
    """A stronger model included as a stretch option / portfolio talking point."""
    model = Sequential(name="BiLSTM_NextWord")
    model.add(Input(shape=(max_len - 1,)))
    model.add(Embedding(total_words, config.EMBEDDING_DIM))
    model.add(Bidirectional(LSTM(config.LSTM_UNITS)))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(Dense(total_words, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model
