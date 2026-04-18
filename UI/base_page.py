import tkinter as tk
from config import BG, SURFACE, ACCENT, TEXT, LARGEFONT, BTN_BACK


class BasePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg=BG)
        self.controller = controller

    def add_title(self, text):
        tk.Label(self, text=text, font=LARGEFONT, bg=BG, fg=ACCENT).place(
            relx=0.5, rely=0.12, anchor="center")

    def add_back_button(self, text, page):
        tk.Button(self, text=f"← {text}", **BTN_BACK,
                  command=lambda: self.controller.show_page(page)).place(x=12, y=12)
