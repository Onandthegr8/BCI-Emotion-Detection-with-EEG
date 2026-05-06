#raw_emotion_training_gru.py
import os
import glob
from pathlib import Path
import numpy as np
import joblib
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DATA_DIR = Path("raw_data")                     # subfolders: NEGATIVE, NEUTRAL, POSITIVE
SUBFOLDERS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL_MAP = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

TARGET_SFREQ = 256                              # Hz after resampling
TARGET_LEN = 2548                               # samples per recording
RANDOM_STATE = 42

MODEL_DIR = Path("raw_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "raw_emotion_gru_model.h5"
SCALER_PATH = MODEL_DIR / "raw_gru_scaler.pkl"


# ---------------------------------------------------
# EDF PREPROCESSING
# ---------------------------------------------------
def load_and_preprocess_edf(path):
    """Load EDF, filter, resample, pad/trim, return (timesteps, channels)."""

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

    # Pick EEG channels only (fallback: all)
    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        picks = np.arange(len(raw.ch_names))

    raw.pick(picks)

    # Resample
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, npad="auto")

    # Filtering
    if raw.n_times < 500:
        raw.filter(1.0, 30.0, method="iir", verbose=False)
        raw.notch_filter(50.0, method="iir", verbose=False)
    else:
        raw.filter(1.0, 30.0, verbose=False)
        raw.notch_filter(50.0, verbose=False)

    data = raw.get_data()  # (channels, samples)
    n_channels, n_times = data.shape

    # Trim / pad
    if n_times >= TARGET_LEN:
        start = (n_times - TARGET_LEN) // 2
        data = data[:, start:start + TARGET_LEN]
    else:
        pad_width = TARGET_LEN - n_times
        data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")

    return data.T  # → (timesteps, channels)


# ---------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------
def build_dataset():
    X_list, y_list = [], []
    total = 0

    for label in SUBFOLDERS:
        folder = DATA_DIR / label
        edf_files = sorted(glob.glob(str(folder / "*.edf")))
        print(f"{label}: {len(edf_files)} files")

        for p in edf_files:
            try:
                arr = load_and_preprocess_edf(p)
                X_list.append(arr)
                y_list.append(LABEL_MAP[label])
                total += 1
            except Exception as e:
                print(f"Error with {p}: {e}")

    if total == 0:
        raise RuntimeError("No EDF files detected!")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    print(f"Dataset: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    print("Loading dataset...")
    X, y = build_dataset()

    # Unify channel count (if mismatch)
    max_channels = max(x.shape[1] for x in X)
    if any(x.shape[1] != max_channels for x in X):
        X_fixed = np.zeros((X.shape[0], TARGET_LEN, max_channels))
        for i in range(X.shape[0]):
            ch = X[i].shape[1]
            X_fixed[i, :, :ch] = X[i]
        X = X_fixed
        print("Unified channels →", max_channels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scaling
    ns, ts, nch = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(ns * ts, nch))
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved →", SCALER_PATH)

    X_train_scaled = scaler.transform(X_train.reshape(ns * ts, nch)).reshape(ns, ts, nch)
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0] * ts, nch)).reshape(X_test.shape[0], ts, nch)

    # ---------------------------------------------------
    # PURE GRU MODEL (no LSTM)
    # ---------------------------------------------------
    inputs = tf.keras.Input(shape=(ts, nch))

    x = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.GRU(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True)
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
    print("Saved GRU model →", MODEL_PATH)

    # ---------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=SUBFOLDERS))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g",
                xticklabels=SUBFOLDERS, yticklabels=SUBFOLDERS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png")
    print("Confusion matrix saved →", MODEL_DIR / "confusion_matrix_gru.png")
