#raw_emotion_training_rnn.py
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
DATA_DIR = Path("raw_data")                    # folder with subfolders NEGATIVE, NEUTRAL, POSITIVE
SUBFOLDERS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL_MAP = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

TARGET_SFREQ = 256        # Hz target sampling rate (resample all edf to this)
TARGET_LEN = 2548         # number of time samples per segment expected by model
RANDOM_STATE = 42
MODEL_DIR = Path("raw_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "raw_emotion_rnn_model.h5"
SCALER_PATH = MODEL_DIR / "raw_scaler.pkl"

# ---------------------------
# UTILITIES
# ---------------------------
def load_and_preprocess_edf(path, target_sfreq=TARGET_SFREQ, target_len=TARGET_LEN):
    """
    Loads EDF, extracts EEG channels, resamples, bandpass + notch filters,
    then trims/pads to target_len. Returns array shape (timesteps, n_channels).
    """
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    # pick EEG channels only (if none, fallback to all channels)
    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        picks = np.arange(len(raw.ch_names))
    raw.pick(picks)

    # ensure sampling frequency
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, npad="auto")

    # safe filtering: use IIR if very short; otherwise default (FIR)
    if raw.n_times < 500:
        raw.filter(1.0, 30.0, method="iir", verbose=False)
        raw.notch_filter(50.0, method="iir", verbose=False)
    else:
        raw.filter(1.0, 30.0, verbose=False)
        raw.notch_filter(50.0, verbose=False)

    data = raw.get_data()   # shape (n_channels, n_times)
    n_channels, n_times = data.shape

    # trim or pad each channel to target_len
    if n_times >= target_len:
        # take center window for better representativeness
        start = (n_times - target_len) // 2
        data = data[:, start:start + target_len]
    else:
        # pad with zeros at end
        pad_width = target_len - n_times
        data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    # transpose to (timesteps, channels)
    return data.T  # shape (target_len, n_channels)


def build_dataset(data_dir=DATA_DIR):
    X_list = []
    y_list = []
    files_count = 0

    for label_name in SUBFOLDERS:
        folder = data_dir / label_name
        if not folder.exists():
            print(f"Warning: folder not found: {folder} (skipping)")
            continue
        edf_paths = sorted(glob.glob(str(folder / "*.edf")))
        print(f"Found {len(edf_paths)} files in {label_name}")
        for p in edf_paths:
            try:
                arr = load_and_preprocess_edf(p)
                # arr shape: (timesteps, n_channels)
                X_list.append(arr)
                y_list.append(LABEL_MAP[label_name])
                files_count += 1
            except Exception as e:
                print(f"Error reading {p}: {e}")

    if files_count == 0:
        raise RuntimeError("No EDF files loaded. Check DATA_DIR and subfolders.")
    X = np.stack(X_list, axis=0)  # (n_samples, timesteps, n_channels)
    y = np.array(y_list, dtype=int)
    print(f"Built dataset X.shape={X.shape}, y.shape={y.shape}")
    return X, y


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("Building dataset from EDF folders...")
    X, y = build_dataset()

    # If channels vary between files, unify to max channels by padding channels dimension
    # (rare if all EDFs have same montage; this code handles mismatches)
    max_ch = max(x.shape[1] for x in X)
    if any(x.shape[1] != max_ch for x in X):
        X_fixed = np.zeros((X.shape[0], TARGET_LEN, max_ch), dtype=float)
        for i in range(X.shape[0]):
            ch = X[i].shape[1]
            X_fixed[i, :, :ch] = X[i]
        X = X_fixed
        print("Unified channel count to", max_ch)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale per-channel using training data: reshape to (n_samples*timesteps, n_channels)
    ns, ts, nch = X_train.shape
    reshaped = X_train.reshape(ns * ts, nch)
    scaler = StandardScaler()
    scaler.fit(reshaped)

    # apply scaler
    X_train_scaled = scaler.transform(reshaped).reshape(ns, ts, nch)
    nt = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(nt * ts, nch)).reshape(nt, ts, nch)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print("Saved scaler to", SCALER_PATH)

    # Build GRU RNN model
    timesteps = X_train_scaled.shape[1]
    channels = X_train_scaled.shape[2]
    n_classes = len(SUBFOLDERS)

    inputs = tf.keras.Input(shape=(timesteps, channels))
    x = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.GRU(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # Callbacks
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

    # Save final model (ModelCheckpoint will already have saved best)
    model.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=LABEL_MAP.keys()))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="g", xticklabels=LABEL_MAP.keys(), yticklabels=LABEL_MAP.keys(), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png")
    print("Saved confusion matrix to", MODEL_DIR / "confusion_matrix.png")
