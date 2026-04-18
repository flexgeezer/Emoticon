import os
import tkinter as tk
from config import BG, SURFACE, ACCENT, TEXT, MUTED, LARGEFONT, SUBTITLEFONT, BTN_SMALL, EMOTIVE_DATA_DIR
from UI.base_page import BasePage
from UI.helpers import get_file_name


class ReportHistory(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.add_back_button("Main Menu", 'StartPage')

        tk.Label(self, text="Report History", font=LARGEFONT,
                 bg=BG, fg=ACCENT).place(relx=0.5, rely=0.1, anchor="center")

        tk.Button(self, text="↻  Refresh", bg=SURFACE, fg=TEXT, relief="flat",
                  font=("Verdana", 9), padx=10, pady=5, cursor="hand2",
                  activebackground=MUTED, activeforeground=TEXT,
                  command=self.refresh).place(relx=1.0, x=-12, y=12, anchor="ne")

        self._list_frame = tk.Frame(self, bg=BG)
        self._list_frame.place(relx=0.5, rely=0.55, anchor="center", relwidth=0.5, relheight=0.72)

        self._canvas = tk.Canvas(self._list_frame, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(self._list_frame, orient=tk.VERTICAL,
                                 command=self._canvas.yview, bg=SURFACE, troughcolor=BG)

        self._canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = tk.Frame(self._canvas, bg=BG)
        self._canvas.create_window((0, 0), window=self._inner, anchor=tk.NW)
        self._canvas.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))

        self._populate()

    def _populate(self):
        os.makedirs(EMOTIVE_DATA_DIR, exist_ok=True)
        for widget in self._inner.winfo_children():
            widget.destroy()
        for fname in os.listdir(EMOTIVE_DATA_DIR):
            full = os.path.join(EMOTIVE_DATA_DIR, fname)
            if os.path.isfile(full):
                tk.Button(self._inner, text=get_file_name(fname), **BTN_SMALL,
                          command=lambda p=full: self.controller.load_report(p)).pack(
                              padx=10, pady=6)

    def refresh(self):
        self._populate()
