import tkinter as tk
from Modes import Live_Mode
from config import BG, ACCENT, MUTED, LARGEFONT, SUBTITLEFONT, BTN_PRIMARY
from UI.base_page import BasePage


class ModeSelectPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.add_back_button("Main Menu", 'StartPage')

        center = tk.Frame(self, bg=BG)
        center.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(center, text="Select Mode", font=LARGEFONT, bg=BG, fg=ACCENT).pack(pady=(0, 6))
        tk.Label(center, text="choose how you want to use Emoticon",
                 font=SUBTITLEFONT, bg=BG, fg=MUTED).pack(pady=(0, 32))

        tk.Button(center, text="Live Mode", **BTN_PRIMARY,
                  command=lambda: Live_Mode()).pack(pady=8, fill='x')
        tk.Button(center, text="Diagnosis Mode", **BTN_PRIMARY,
                  command=lambda: controller.show_page('InputSelect')).pack(pady=8, fill='x')
