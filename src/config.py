"""
config.py
---------
Central configuration for the LexiPredict Next-Word Prediction project.
Keeping paths and hyperparameters in one place avoids magic numbers/strings
scattered across the codebase (an industry best practice).
"""

import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")

RNN_MODEL_PATH = os.path.join(MODEL_DIR, "rnn_model.keras")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
BILSTM_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_model.keras")

TOKENIZER_PATH = os.path.join(ARTIFACT_DIR, "tokenizer.pkl")
MAX_LEN_PATH = os.path.join(ARTIFACT_DIR, "max_len.pkl")
HISTORY_PATH = os.path.join(ARTIFACT_DIR, "training_history.json")

# ---------------------------------------------------------------------------
# DATA / TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
RNN_UNITS = 96
LSTM_UNITS = 128
DROPOUT_RATE = 0.3

VALIDATION_SPLIT = 0.15
BATCH_SIZE = 16
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 0.001

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE = 0.7
DEFAULT_WORDS_TO_GENERATE = 5
