import tkinter as tk
from pages import StartPage
from pages import ModeSelectPage
from pages import Settings
from pages import InputSelect
from reportHistory import ReportHistory

class EmoticonApp(tk.Tk):

    # __init__ function for class EmoticonApp
    def __init__(self):
        # __init__ function for class Tk
        tk.Tk.__init__(self)
        self.geometry("1000x500")
        self.title("Emoticon")
        # creating a container
        x = 5
        y = 5
        container = tk.Frame(self)
        container.pack(side="top", fill="both", anchor="center", expand=True)

        container.grid_rowconfigure(0, weight=1)

        container.grid_columnconfigure(0, weight=1)

        # initialising frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, ModeSelectPage, Settings,ReportHistory,InputSelect):
            frame = F(container, self)

            # initialising frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame
            # resizes grid of initialised frames to spread evenly to fit size of window
            for i in range(x):
                frame.grid_rowconfigure(i, weight=1)
            for i in range(y):
                frame.grid_columnconfigure(i, weight=1)
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_page(StartPage)

    # to display the current frame passed as
    # parameter
    def show_page(self, page):
        frame = self.frames[page]
        frame.tkraise()
