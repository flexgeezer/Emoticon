import tkinter as tk
from config import BG
from .start import StartPage
from .mode_select import ModeSelectPage
from .report_history import ReportHistory
from .input_select import InputSelect
from .report_page import Report


class EmoticonApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("1000x560")
        self.title("Emoticon")
        self.configure(bg=BG)
        self.resizable(False, False)

        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, ModeSelectPage, ReportHistory, InputSelect, Report):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_page(StartPage)

    def show_page(self, page):
        if isinstance(page, str):
            page = next(k for k in self.frames if k.__name__ == page)
        self.frames[page].tkraise()

    def load_report(self, file_path):
        self.frames[Report].load(file_path)
        self.show_page(Report)
