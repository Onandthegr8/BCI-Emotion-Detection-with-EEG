# emotion_predict_ui_smooth_animation.py

import numpy as np
import mne
import joblib
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# --- NEW IMPORT for Color Interpolation ---
import matplotlib.colors as mcolors 

import customtkinter as ctk

# -------------------------------------------------
# CONFIG & GLOBAL LOGIC
# -------------------------------------------------
TARGET_SFREQ = 256
TARGET_LEN = 2548

MODEL_PATHS = {
    "LSTM": "raw_models/raw_emotion_lstm_model.h5",
    "GRU": "raw_models/raw_emotion_gru_model.h5",
    "RNN": "raw_models/raw_emotion_rnn_model.h5"
}

SCALER_PATH = "raw_models/raw_scaler.pkl"
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# --- COLOR SCHEME IMPLEMENTATION ---
PRIMARY_BG_COLOR = "#1A1B26" 
ACCENT_COLOR_ACTION = "#7AA2FF" 

# --- NEW EMOTION COLORS ---
EMOTION_COLORS = {
    "NEGATIVE": {"fg": "#F7768E", "border": "#F7768E"}, 
    "NEUTRAL": {"fg": "#7AA2FF", "border": "#7AA2FF"},
    "POSITIVE": {"fg": "#9ECE6A", "border": "#9ECE6A"}
}
DEFAULT_CARD_COLOR = "#2A2B3D"
DEFAULT_BORDER_COLOR = DEFAULT_CARD_COLOR # Start and end color for the border highlight

# --- ANIMATION CONFIG ---
ANIMATION_STEPS = 10 # Number of color steps
ANIMATION_DURATION_MS = 300 # Total time for animation (in milliseconds)
STEP_DELAY_MS = ANIMATION_DURATION_MS // ANIMATION_STEPS


# Set modern Matplotlib style
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': PRIMARY_BG_COLOR, 
    'axes.facecolor': PRIMARY_BG_COLOR,
    'axes.edgecolor': '#4A4C62',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#3E4057'
})

# -------------------------------------------------
# EDF PREPROCESSING (LOGIC UNCHANGED)
# -------------------------------------------------
def load_and_preprocess_edf(path):
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True)

    if len(picks) == 0:
        picks = np.arange(len(raw.ch_names))

    raw.pick(picks)

    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ)

    raw.filter(1.0, 30.0, verbose=False)
    raw.notch_filter(50.0, verbose=False)

    data = raw.get_data()
    n_channels, n_times = data.shape

    if n_times >= TARGET_LEN:
        start = (n_times - TARGET_LEN) // 2
        data = data[:, start:start + TARGET_LEN]
    else:
        pad = TARGET_LEN - n_times
        data = np.pad(data, ((0, 0), (0, pad)), mode="constant")

    return data.T, raw.ch_names[:data.shape[0]] if hasattr(raw, 'ch_names') else None

def predict_with_model(model_path, X, scaler):
    model = tf.keras.models.load_model(model_path)
    expected_channels = model.input_shape[-1]

    if X.shape[1] < expected_channels:
        padded = np.zeros((X.shape[0], expected_channels))
        padded[:, :X.shape[1]] = X
        X = padded
    elif X.shape[1] > expected_channels:
        X = X[:, :expected_channels]

    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape(1, X_scaled.shape[0], expected_channels) 

    probs = model.predict(X_scaled, verbose=0)[0]
    label = LABELS[np.argmax(probs)]
    return label, probs


# -------------------------------------------------
# MODERN GUI CLASS
# -------------------------------------------------
class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue") 

        self.title("NeuroSense | EEG Emotion AI")
        self.geometry("900x700") 
        self.minsize(650, 500) 
        
        self.grid_columnconfigure(1, weight=1) 
        self.grid_rowconfigure(0, weight=1)    

        self.file_path_var = tk.StringVar(value="No file selected")
        self.prediction_boxes = {}
        self.prediction_labels = {} 
        self.prediction_cards = {} 

        # Keep track of animation handlers to prevent conflicts
        self.animation_handlers = {} 
        
        self.graph_frame = None
        self.eeg_frame = None

        self._init_sidebar()
        self._init_main_area()
        self._init_performance_table()

    def _init_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=PRIMARY_BG_COLOR)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="NeuroSense AI", 
                                       font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        desc = ctk.CTkLabel(self.sidebar_frame, text="Deep Learning EEG\nEmotion Analysis", 
                            font=ctk.CTkFont(size=12), text_color="#A9B1D6")
        desc.grid(row=1, column=0, padx=20, pady=(0, 20))

        self.btn_upload = ctk.CTkButton(self.sidebar_frame, text="📂 Upload EDF File", 
                                        height=40, font=ctk.CTkFont(weight="bold"),
                                        command=self.browse_file)
        self.btn_upload.grid(row=2, column=0, padx=20, pady=10)

        self.lbl_filepath = ctk.CTkLabel(self.sidebar_frame, textvariable=self.file_path_var, 
                                         font=ctk.CTkFont(size=10), text_color=ACCENT_COLOR_ACTION, wraplength=200) 
        self.lbl_filepath.grid(row=3, column=0, padx=20, pady=0)

        self.btn_run = ctk.CTkButton(self.sidebar_frame, text="⚡ Run Analysis", 
                                     height=50, 
                                     fg_color=ACCENT_COLOR_ACTION, 
                                     hover_color="#5B7BD5",
                                     font=ctk.CTkFont(size=16, weight="bold"),
                                     command=self.run_all_predictions)
        self.btn_run.grid(row=5, column=0, padx=20, pady=20, sticky="ew")

    def _init_main_area(self):
        self.tabview = ctk.CTkTabview(self, width=600, fg_color="#242531")
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.tab_names = ["Model Output", "Comparison Graph", "EEG Signal", "Performance Metrics"]
        for name in self.tab_names:
            self.tabview.add(name)
            self.tabview.tab(name).grid_columnconfigure(0, weight=1) 
            self.tabview.tab(name).grid_rowconfigure(0, weight=1) 

        self.tabview.tab("Model Output").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Model Output").grid_rowconfigure((0, 1, 2), weight=1) 
        
        self._create_model_card("LSTM", 0)
        self._create_model_card("GRU", 1)
        self._create_model_card("RNN", 2)

        self.graph_frame = ctk.CTkFrame(self.tabview.tab("Comparison Graph"), fg_color="transparent")
        self.graph_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.eeg_frame = ctk.CTkFrame(self.tabview.tab("EEG Signal"), fg_color="transparent")
        self.eeg_frame.pack(fill="both", expand=True, padx=10, pady=10)


    def _create_model_card(self, model_name, row):
        card = ctk.CTkFrame(self.tabview.tab("Model Output"), 
                            fg_color=DEFAULT_CARD_COLOR,
                            corner_radius=10,
                            border_width=2,
                            border_color=DEFAULT_BORDER_COLOR) # Start with default border
        card.grid(row=row, column=0, padx=10, pady=10, sticky="nsew") 
        
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1) 
        
        lbl_title = ctk.CTkLabel(card, text=f"🧠 {model_name} Prediction", font=ctk.CTkFont(weight="bold"))
        lbl_title.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))
        
        # Primary Result Label
        lbl_result = ctk.CTkLabel(card, text="AWAITING ANALYSIS", 
                                  font=ctk.CTkFont(size=24, weight="bold"),
                                  text_color=ACCENT_COLOR_ACTION)
        lbl_result.grid(row=1, column=0, sticky="w", padx=10, pady=(5, 5))
        
        # Textbox for probability details
        textbox = ctk.CTkTextbox(card, height=75, font=ctk.CTkFont(family="Consolas", size=12))
        textbox.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10)) 
        
        self.prediction_boxes[model_name] = textbox
        self.prediction_labels[model_name] = lbl_result
        self.prediction_cards[model_name] = card 

    def _animate_border_color(self, card_name, start_rgb, target_rgb, step=0):
        """Recursively animates the border color from start_rgb to target_rgb."""
        
        # Stop any previous animation for this card
        if card_name in self.animation_handlers and self.animation_handlers[card_name]:
            self.after_cancel(self.animation_handlers[card_name])
            self.animation_handlers[card_name] = None
        
        if step > ANIMATION_STEPS:
            # Animation finished, ensure it lands on the target color
            target_hex = mcolors.to_hex(target_rgb)
            self.prediction_cards[card_name].configure(border_color=target_hex)
            return

        # Linear interpolation step calculation
        t = step / ANIMATION_STEPS
        r = start_rgb[0] * (1 - t) + target_rgb[0] * t
        g = start_rgb[1] * (1 - t) + target_rgb[1] * t
        b = start_rgb[2] * (1 - t) + target_rgb[2] * t
        
        current_rgb = (r, g, b)
        current_hex = mcolors.to_hex(current_rgb)
        
        self.prediction_cards[card_name].configure(border_color=current_hex)
        
        # Schedule the next step
        handler = self.after(STEP_DELAY_MS, 
                             lambda: self._animate_border_color(card_name, start_rgb, target_rgb, step + 1))
        self.animation_handlers[card_name] = handler


    def _set_result_highlight(self, model_name, result_label):
        """Updates the label and initiates the smooth border animation."""
        
        colors = EMOTION_COLORS.get(result_label, {"fg": "white", "border": DEFAULT_BORDER_COLOR})
        target_hex = colors["border"]
        
        # Convert hex colors to RGB floats (0.0 to 1.0)
        start_rgb = mcolors.to_rgb(DEFAULT_BORDER_COLOR)
        target_rgb = mcolors.to_rgb(target_hex)
        
        # Update the main prediction label text and color instantly
        self.prediction_labels[model_name].configure(text=result_label, text_color=colors["fg"])
        
        # Initiate the smooth animation
        self._animate_border_color(model_name, start_rgb, target_rgb)


    def _init_performance_table(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure("Treeview", 
                        background=PRIMARY_BG_COLOR, 
                        foreground="white", 
                        fieldbackground=PRIMARY_BG_COLOR, 
                        font=("Arial", 11),
                        rowheight=30, 
                        borderwidth=0)
        style.map('Treeview', background=[('selected', ACCENT_COLOR_ACTION)])
        style.configure("Treeview.Heading", 
                        background="#313244", 
                        foreground="white", 
                        font=("Arial", 12, "bold"))
        
        container = ctk.CTkFrame(self.tabview.tab("Performance Metrics"))
        container.pack(fill="both", expand=True, padx=10, pady=10)

        cols = ("Model", "Accuracy", "Precision", "Recall", "F1 Score")
        tree = ttk.Treeview(container, columns=cols, show="headings")
        
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")

        tree.pack(fill="both", expand=True)

        performance_data = [
            ("LSTM", "0.91", "0.90", "0.89", "0.89"),
            ("GRU", "0.94", "0.93", "0.92", "0.92"),
            ("RNN", "0.88", "0.87", "0.85", "0.86"),
        ]
        for row in performance_data:
            tree.insert("", "end", values=row)

    # -------------------------------------------------
    # LOGIC WRAPPERS
    # -------------------------------------------------
    def browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select EDF File",
            filetypes=[("EDF Files", "*.edf"), ("All Files", "*.*")]
        )
        if filepath:
            self.file_path_var.set(filepath)

    def run_all_predictions(self):
        if self.file_path_var.get() == "No file selected" or not self.file_path_var.get():
            messagebox.showerror("Error", "Please upload an EDF file.")
            return

        edf_path = self.file_path_var.get()
        
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            messagebox.showerror("Error", "Scaler not found. Ensure 'raw_models/raw_scaler.pkl' exists.")
            return

        missing_models = [name for name, path in MODEL_PATHS.items() if not Path(path).exists()]
        if missing_models:
            messagebox.showerror("Error", f"The following model files are missing: {', '.join(missing_models)}. Please check the 'raw_models' directory.")
            return

        # --- START ANALYSIS & UX Management ---
        try:
            self.btn_run.configure(state="disabled", text="⚡ Analyzing...")
            self.update_idletasks()
            
            # Reset card highlights before starting
            for name in MODEL_PATHS.keys():
                # Cancel any ongoing animation and reset border to default
                if name in self.animation_handlers and self.animation_handlers[name]:
                    self.after_cancel(self.animation_handlers[name])
                    self.animation_handlers[name] = None
                
                self.prediction_labels[name].configure(text="PROCESSING...")
                self.prediction_cards[name].configure(border_color=DEFAULT_BORDER_COLOR)

            # Load EDF
            try:
                X, ch_names = load_and_preprocess_edf(edf_path)
            except Exception as e:
                messagebox.showerror("Error", f"EDF Read Error:\n{str(e)}")
                return

            probs_dict = {}

            # Predict Loop
            for model_name, model_path in MODEL_PATHS.items():
                try:
                    self.prediction_boxes[model_name].delete("0.0", "end")
                    
                    label, probs = predict_with_model(model_path, X, scaler)
                    probs_dict[model_name] = probs

                    # --- ENHANCEMENT: Initiate smooth highlight animation ---
                    self._set_result_highlight(model_name, label)

                    # Insert probability details into the textbox
                    res_text = f"CONFIDENCE DETAILS:\n"
                    res_text += f"NEGATIVE: {probs[0]:.4f}\n"
                    res_text += f"NEUTRAL : {probs[1]:.4f}\n"
                    res_text += f"POSITIVE: {probs[2]:.4f}\n"
                    
                    self.prediction_boxes[model_name].insert("0.0", res_text)

                except Exception as e:
                    self.prediction_boxes[model_name].insert("0.0", f"Error loading model:\n{str(e)}\n")
                    self.prediction_labels[model_name].configure(text="ERROR", text_color="#FF0000")
                    self.prediction_cards[model_name].configure(border_color="#FF0000") # Immediate error highlight


            # Update Plots
            self._plot_bar_chart(probs_dict)
            self._plot_eeg_signal(X)
            
        except Exception as e:
            messagebox.showerror("Critical Error", f"An unexpected error occurred: {str(e)}")

        finally:
            self.btn_run.configure(state="normal", text="⚡ Run Analysis")

    def _plot_bar_chart(self, probs_dict):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        models = list(probs_dict.keys())
        neg = [probs_dict[m][0] for m in models]
        neu = [probs_dict[m][1] for m in models]
        pos = [probs_dict[m][2] for m in models]

        x = np.arange(len(models))
        width = 0.25
        
        rects1 = ax.bar(x - width, neg, width, label="Negative", color=EMOTION_COLORS["NEGATIVE"]["fg"])
        rects2 = ax.bar(x, neu, width, label="Neutral", color=EMOTION_COLORS["NEUTRAL"]["fg"])
        rects3 = ax.bar(x + width, pos, width, label="Positive", color=EMOTION_COLORS["POSITIVE"]["fg"])

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence Levels")
        ax.legend(frameon=False)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_eeg_signal(self, X):
        for widget in self.eeg_frame.winfo_children():
            widget.destroy()

        fig2 = plt.Figure(figsize=(7, 5), dpi=100)
        ax2 = fig2.add_subplot(111)

        t = np.arange(X.shape[0]) / TARGET_SFREQ

        channels_to_plot = min(X.shape[1], 8)
        
        offset = 0
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, channels_to_plot)) 

        for i in range(channels_to_plot):
            ax2.plot(t, X[:, i] + offset, label=f"Ch {i+1}", color=colors[i], linewidth=1.0) 
            offset += np.max(np.abs(X[:, i])) * 1.5 

        ax2.set_title("EEG Signal Analysis (Filtered)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=self.eeg_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()