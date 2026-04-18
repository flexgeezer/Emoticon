import sys
import tkinter as tk
from config import BG, ACCENT, TEXT, MUTED, LARGEFONT, SUBTITLEFONT, BTN_PRIMARY, BTN_DANGER
from UI.base_page import BasePage


class StartPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)

        center = tk.Frame(self, bg=BG)
        center.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(center, text="Emoticon", font=LARGEFONT, bg=BG, fg=ACCENT).pack(pady=(0, 4))
        tk.Label(center, text="real-time emotion recognition", font=SUBTITLEFONT,
                 bg=BG, fg=MUTED).pack(pady=(0, 32))

        tk.Button(center, text="Start Session", **BTN_PRIMARY,
                  command=lambda: controller.show_page('ModeSelectPage')).pack(pady=8, fill='x')
        tk.Button(center, text="Report History", **BTN_PRIMARY,
                  command=lambda: controller.show_page('ReportHistory')).pack(pady=8, fill='x')

        tk.Button(self, text="Exit", **BTN_DANGER,
                  command=sys.exit).place(relx=1.0, x=-12, y=12, anchor="ne")
