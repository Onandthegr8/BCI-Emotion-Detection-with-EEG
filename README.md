# Brain-Computer Interface — EEG Emotion Detection

A deep learning system that reads raw EEG recordings (`.edf` files) and classifies the emotional state of the subject into **Negative**, **Neutral**, or **Positive** using three recurrent neural network architectures — LSTM, GRU, and RNN — with a dark-themed desktop GUI for real-time prediction.

---

## Demo

| Model Output | Comparison Graph | EEG Signal |
|---|---|---|
| Color-coded prediction cards per model | Side-by-side probability bar chart | Filtered multi-channel EEG waveform |

> The GUI is built with `customtkinter` and named **NeuroSense AI**.

---

## How It Works

```
Raw EEG (.edf)
     │
     ▼
MNE Preprocessing
  • Pick EEG channels
  • Resample → 256 Hz
  • Bandpass filter: 1–30 Hz
  • Notch filter: 50 Hz (power-line noise)
  • Trim / pad → 2548 samples (~10 s)
     │
     ▼
StandardScaler  (fit on training data, saved as raw_scaler.pkl)
     │
     ▼
┌──────────┐  ┌─────────┐  ┌─────────┐
│  LSTM    │  │   GRU   │  │   RNN   │
│  model   │  │  model  │  │  model  │
└────┬─────┘  └────┬────┘  └────┬────┘
     └──────────────┴────────────┘
                   │
                   ▼
        NEGATIVE / NEUTRAL / POSITIVE
```

---

## Model Architectures

All three models share the same two-layer recurrent pattern:

| Layer | LSTM | GRU | RNN |
|---|---|---|---|
| Recurrent 1 | LSTM(128) | GRU(128) | GRU(128) |
| Dropout | 0.3 | 0.3 | 0.3 |
| Recurrent 2 | LSTM(64) | GRU(64) | GRU(64) |
| Dropout | 0.3 | 0.3 | 0.3 |
| Dense | 64, ReLU | 64, ReLU | 64, ReLU |
| Output | 3, Softmax | 3, Softmax | 3, Softmax |

**Training config:** Adam optimizer · sparse categorical cross-entropy · 60 epochs max · batch size 16 · early stopping (patience = 6) · 80/20 train-test split

---

## Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| LSTM | 91% | 0.90 | 0.89 | 0.89 |
| **GRU** | **94%** | **0.93** | **0.92** | **0.92** |
| RNN | 88% | 0.87 | 0.85 | 0.86 |

---

## Project Structure

```
├── raw_emotion_training_lstm.py   # Train LSTM model
├── raw_emotion_training_gru.py    # Train GRU model
├── raw_emotion_training_rnn.py    # Train RNN model
├── predict_all.py                 # Desktop GUI — loads .edf, runs all 3 models
│
├── raw_data/                      # ← place your EDF files here
│   ├── NEGATIVE/                  #   .edf files labelled negative emotion
│   ├── NEUTRAL/                   #   .edf files labelled neutral emotion
│   └── POSITIVE/                  #   .edf files labelled positive emotion
│
└── raw_models/                    # Generated after training
    ├── raw_emotion_lstm_model.h5
    ├── raw_emotion_gru_model.h5
    ├── raw_emotion_rnn_model.h5
    └── raw_scaler.pkl
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- MNE-Python
- scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn
- customtkinter

Install all dependencies:

```bash
pip install tensorflow mne scikit-learn numpy pandas matplotlib seaborn customtkinter joblib
```

---

## Quickstart

### 1. Prepare your data

Place your EEG recordings inside `raw_data/` split by emotion label:

```
raw_data/
  NEGATIVE/   ← .edf files
  NEUTRAL/    ← .edf files
  POSITIVE/   ← .edf files
```

### 2. Train the models

Run each training script (you can run them independently):

```bash
python raw_emotion_training_lstm.py
python raw_emotion_training_gru.py
python raw_emotion_training_rnn.py
```

Each script will:
- Load and preprocess all EDF files
- Train the model with early stopping
- Save the model to `raw_models/`
- Print a classification report and save a confusion matrix PNG

### 3. Run the GUI

```bash
python predict_all.py
```

In the **NeuroSense AI** window:
1. Click **Upload EDF File** and select any `.edf` recording
2. Click **Run Analysis**
3. View results across four tabs:
   - **Model Output** — prediction cards for LSTM, GRU, and RNN with animated color highlights
   - **Comparison Graph** — side-by-side confidence bar chart
   - **EEG Signal** — filtered multi-channel waveform
   - **Performance Metrics** — model accuracy/precision/recall/F1 table

---

## EEG Preprocessing Details

| Step | Parameter |
|---|---|
| Channel selection | EEG channels only (fallback: all channels) |
| Target sampling rate | 256 Hz |
| Bandpass filter | 1–30 Hz (FIR; IIR for very short recordings) |
| Notch filter | 50 Hz (power-line interference) |
| Segment length | 2548 samples (center-crop or zero-pad) |

---

## Dataset

This project expects EEG data in **EDF (European Data Format)**. Any publicly available EEG emotion dataset in EDF format can be used, for example:

- [DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- [SEED Dataset](https://bcmi.sjtu.edu.cn/~seed/seed.html)
- [MAHNOB-HCI](https://mahnob-db.eu/hci-tagging/)

> The dataset itself is not included in this repository. You must download it separately and arrange it under `raw_data/` as described above.

---

## Tech Stack

| Component | Library |
|---|---|
| EEG I/O & preprocessing | `mne` |
| Deep learning models | `tensorflow` / `keras` |
| Feature scaling | `scikit-learn` |
| Desktop GUI | `customtkinter` + `tkinter` |
| Visualization | `matplotlib`, `seaborn` |
| Model persistence | `joblib` |
