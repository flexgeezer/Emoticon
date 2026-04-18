import tkinter as tk
from tkinter import filedialog
from Modes import Diagnosis_Mode
from config import BG, ACCENT, MUTED, LARGEFONT, SUBTITLEFONT, BTN_PRIMARY
from UI.base_page import BasePage


class InputSelect(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.add_back_button("Mode Select", 'ModeSelectPage')

        center = tk.Frame(self, bg=BG)
        center.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(center, text="Choose Input", font=LARGEFONT, bg=BG, fg=ACCENT).pack(pady=(0, 6))
        tk.Label(center, text="select your video source for the session",
                 font=SUBTITLEFONT, bg=BG, fg=MUTED).pack(pady=(0, 32))

        tk.Button(center, text="Webcam", **BTN_PRIMARY,
                  command=self.start_diagnosis).pack(pady=8, fill='x')
        tk.Button(center, text="Video from Device", **BTN_PRIMARY,
                  command=self.select_video).pack(pady=8, fill='x')

    def start_diagnosis(self, video_source=0):
        mode = Diagnosis_Mode(video_source)
        if mode.emoticon_file:
            self.controller.load_report(mode.emoticon_file)

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if path:
            self.start_diagnosis(video_source=path)
