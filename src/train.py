"""
train.py
--------
Trains the RNN, LSTM, and BiLSTM models with early stopping and
learning-rate scheduling, then saves the models (Keras native format),
the tokenizer, and a JSON training history used later for reporting.

Usage:
    python -m src.train
"""

import json
import os
import time

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src import config
from src.data_preprocessing import prepare_data
from src.model import build_rnn_model, build_lstm_model, build_bilstm_model


def get_callbacks():
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]


def train_one(name, build_fn, data, save_path):
    print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")
    model = build_fn(data["total_words"], data["max_len"])
    model.summary()

    start = time.time()
    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=2,
    )
    elapsed = time.time() - start

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(save_path)
    print(f"Saved {name} to {save_path} ({elapsed:.1f}s)")

    return {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "epochs_run": len(history.history["loss"]),
        "training_seconds": round(elapsed, 1),
    }


def main():
    data = prepare_data()
    print(f"Vocabulary size: {data['total_words']} | Max sequence length: {data['max_len']}")

    results = {}
    results["rnn"] = train_one("SimpleRNN", build_rnn_model, data, config.RNN_MODEL_PATH)
    results["lstm"] = train_one("LSTM", build_lstm_model, data, config.LSTM_MODEL_PATH)
    results["bilstm"] = train_one("BiLSTM", build_bilstm_model, data, config.BILSTM_MODEL_PATH)

    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nFinal results summary:")
    for name, r in results.items():
        print(f"  {name:8s} | val_accuracy={r['val_accuracy']:.3f} | val_loss={r['val_loss']:.3f}")


if __name__ == "__main__":
    main()
