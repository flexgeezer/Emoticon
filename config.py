import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTIVE_DATA_DIR = os.path.join(BASE_DIR, "Emotive Data")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")

emoji_dict = {
    0: os.path.join(BASE_DIR, "Emojis", "Emoji0.png"),
    1: os.path.join(BASE_DIR, "Emojis", "Emoji1.png"),
    2: os.path.join(BASE_DIR, "Emojis", "Emoji2.png"),
    3: os.path.join(BASE_DIR, "Emojis", "Emoji3.png"),
}

# Typography
LARGEFONT = ("Verdana", 26, "bold")
SUBTITLEFONT = ("Verdana", 10)
LABELFONT = ("Verdana", 10)

# Color palette
BG      = "#1e1e2e"
SURFACE = "#2a2a3e"
ACCENT  = "#7c9fff"
TEXT    = "#dde1f0"
MUTED   = "#55556d"
DANGER  = "#e06c75"

# Button style presets
BTN_PRIMARY = dict(
    bg=ACCENT, fg=BG, relief="flat",
    font=("Verdana", 11, "bold"), padx=24, pady=12,
    cursor="hand2", activebackground="#9bb3ff", activeforeground=BG,
)
BTN_BACK = dict(
    bg=SURFACE, fg=TEXT, relief="flat",
    font=("Verdana", 9), padx=12, pady=6,
    cursor="hand2", activebackground=MUTED, activeforeground=TEXT,
)
BTN_DANGER = dict(
    bg=DANGER, fg=BG, relief="flat",
    font=("Verdana", 9, "bold"), padx=12, pady=6,
    cursor="hand2", activebackground="#e88891", activeforeground=BG,
)
BTN_SMALL = dict(
    bg=SURFACE, fg=TEXT, relief="flat",
    font=("Verdana", 10), height=2, width=22,
    cursor="hand2", activebackground=ACCENT, activeforeground=BG,
)
BTN_SAVE = dict(
    bg=ACCENT, fg=BG, relief="flat",
    font=("Verdana", 9, "bold"), padx=16, pady=6,
    cursor="hand2", activebackground="#9bb3ff", activeforeground=BG,
)
