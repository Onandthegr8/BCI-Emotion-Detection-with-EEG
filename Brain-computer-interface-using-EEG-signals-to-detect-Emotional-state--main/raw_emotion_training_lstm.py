# train_lstm_from_edf.py
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = Path("raw_data")  # folder with subfolders NEGATIVE, NEUTRAL, POSITIVE
SUBFOLDERS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL_MAP = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

TARGET_SFREQ = 256
TARGET_LEN = 2548
RANDOM_STATE = 42

MODEL_DIR = Path("raw_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "raw_emotion_lstm_model.h5"
SCALER_PATH = MODEL_DIR / "raw_lstm_scaler.pkl"

# ---------------------------
# UTILITIES
# ---------------------------
def load_and_preprocess_edf(path, target_sfreq=256, target_len=2548):
    """Load EDF, filter, resample, normalize length."""
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        picks = np.arange(len(raw.ch_names))

    raw.pick(picks)

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, npad="auto")

    if raw.n_times < 500:
        raw.filter(1.0, 30.0, method="iir", verbose=False)
        raw.notch_filter(50.0, method="iir", verbose=False)
    else:
        raw.filter(1.0, 30.0, verbose=False)
        raw.notch_filter(50.0, verbose=False)

    data = raw.get_data()
    n_channels, n_times = data.shape

    if n_times >= target_len:
        start = (n_times - target_len) // 2
        data = data[:, start:start + target_len]
    else:
        pad = target_len - n_times
        data = np.pad(data, ((0, 0), (0, pad)), mode="constant")

    return data.T  # (timesteps, channels)


def build_dataset(data_dir=DATA_DIR):
    X_list, y_list = [], []
    total_files = 0

    for label in SUBFOLDERS:
        folder = data_dir / label
        edfs = sorted(glob.glob(str(folder / "*.edf")))
        print(f"{label}: {len(edfs)} EDF files")

        for p in edfs:
            try:
                arr = load_and_preprocess_edf(p)
                X_list.append(arr)
                y_list.append(LABEL_MAP[label])
                total_files += 1
            except Exception as e:
                print(f"Error reading {p}: {e}")

    if total_files == 0:
        raise RuntimeError("No EDF files found.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    print(f"Dataset built: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------
# MAIN TRAINING SCRIPT
# ---------------------------
if __name__ == "__main__":
    print("Loading EDF dataset...")
    X, y = build_dataset()

    # unify the channel count if mismatch
    max_ch = max(sample.shape[1] for sample in X)
    if any(sample.shape[1] != max_ch for sample in X):
        print("Padding channels to match maximum:", max_ch)
        new_X = np.zeros((len(X), TARGET_LEN, max_ch))
        for i in range(len(X)):
            ch = X[i].shape[1]
            new_X[i, :, :ch] = X[i]
        X = new_X

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    ns, ts, nch = X_train.shape

    # scale
    reshaped = X_train.reshape(ns * ts, nch)
    scaler = StandardScaler()
    scaler.fit(reshaped)

    X_train_scaled = scaler.transform(reshaped).reshape(ns, ts, nch)
    X_test_scaled = scaler.transform(
        X_test.reshape(len(X_test) * ts, nch)
    ).reshape(len(X_test), ts, nch)

    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved:", SCALER_PATH)

    # ---------------------------
    # LSTM MODEL
    # ---------------------------
    print("Building LSTM Model...")

    timesteps = ts
    channels = nch
    n_classes = len(SUBFOLDERS)

    inputs = tf.keras.Input(shape=(timesteps, channels))

    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy", save_best_only=True
        )
    ]

    # Train
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=16,
        callbacks=callbacks,
        verbose=2
    )

    model.save(MODEL_PATH)
    print("Model saved:", MODEL_PATH)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=LABEL_MAP.keys()))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="g",
        xticklabels=LABEL_MAP.keys(),
        yticklabels=LABEL_MAP.keys(),
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix_lstm.png")

    print("Confusion matrix saved to:",
          MODEL_DIR / "confusion_matrix_lstm.png")
