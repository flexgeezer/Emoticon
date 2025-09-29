import tkinter as tk
# first window frame startpage

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # label of frame Layout 2
        label = tk.Label(self, text="Emoticon", font=LARGEFONT)

        # putting the grid in its place by using
        # grid
        label.grid(row=0, column=2, padx=10, pady=10)

        # button to show Mode Select frame
        button1 = tk.Button(self, text="Mode Select", command=lambda: controller.show_page(Page1))

        # putting the button in its place by
        # using grid
        button1.grid(row=1, column=2, padx=100, pady=10)

        # button to show Report History frame
        button2 = tk.Button(self, text="Report History",
                            command=lambda: controller.show_page(Page2))

        # putting the button in its place by
        # using grid
        button2.grid(row=2, column=2, padx=10, pady=10)

        # button to show Report History frame
        button3 = tk.Button(self, text="Settings",
                            command=lambda: controller.show_page(Page3))

        # putting the button in its place by
        # using grid
        button3.grid(row=3, column=2, padx=10, pady=10)

        # button to show Report History frame
        button4 = tk.Button(self, text="Exit",
                            command=sys.exit)

        # putting the button in its place by
        # using grid
        button4.grid(row=0, column=0, padx=10, pady=10)


# second window frame ModeSelect
class Page1(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Mode Select", font=LARGEFONT)
        label.grid(row=0, column=2, padx=10, pady=10)

        # button to show Startpage frame with text
        # layout2
        button1 = tk.Button(self, text="Main Menu",
                            command=lambda: controller.show_page(StartPage))

        # putting the button in its place
        # by using grid
        button1.grid(row=0, column=0, padx=10, pady=10)

        # button to show Live Mode frame with text
        # layout2
        button2 = tk.Button(self, text="Live Mode",command=lambda: Live_Mode())

        # putting the button in its place
        # by using grid
        button2.grid(row=1, column=2, padx=10, pady=10)

        # button to show Diagnosis frame with text
        # layout2
        button3 = tk.Button(self, text="Diagnosis Mode",
                            command=lambda: controller.show_page(Page4))

        # putting the button in its place
        # by using grid
        button3.grid(row=2, column=2, padx=10, pady=10)

# third window frame ReportHistory
class Page2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        global emoji_dict
        label = tk.Label(self, text="Report History", font=LARGEFONT)
        label.grid(row=0, column=2, padx=10, pady=10,sticky = 'nsew')

        # button to show frame 3 with text
        # layout3
        button1 = tk.Button(self, text="Back to Main Menu",
                            command=lambda: controller.show_page(StartPage))

        # putting the button in its place by
        # using grid
        button1.grid(row=0, column=0, padx=10, pady=10)

        # Create a canvas to hold the buttons
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=1, column=2, padx=10, pady=10,rowspan = 3, sticky="nsew")

        # Add a scrollbar
        scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.grid(row=1, column=3, sticky="ns")

        # Configure the canvas to work with the scrollbar
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas to hold the buttons
        self.inner_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor=tk.NW)

        self.reportbuttons = CSVFileReaderFrame(self.inner_frame)
        self.reportbuttons.pack()

        # Bind the canvas to the function that updates the scrolling region
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        button2 = tk.Button(self, text='Refresh', command=self.refresh)
        button2.grid(row=0, column=3, padx=10, pady=10)

    def refresh(self):
        # Destroy the existing frame
        self.inner_frame.destroy()

        # Create a new frame inside the canvas
        self.inner_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor=tk.NW)

        # Recreate the CSVFileReaderFrame inside the new frame
        self.reportbuttons = CSVFileReaderFrame(self.inner_frame)
        self.reportbuttons.pack()

    def on_canvas_configure(self,event):
        # Update the scrolling region whenever the size of the inner frame changes
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


class CSVFileReaderFrame(tk.Frame):
    def __init__(self, parent, folder_path="/Users/Admin/PycharmProjects/Emoticon/Emotive Data"):
        tk.Frame.__init__(self, parent)
        self.folder_path = folder_path
        self.files = os.listdir(folder_path)
        self.initialise_buttons()
        global emoji_dict

    def initialise_buttons(self):
        for file_name in self.files:
            if os.path.isfile(os.path.join(self.folder_path, file_name)):
                button = tk.Button(self, height=2, width=20, text=self.get_file_name(file_name),
                                   command=lambda path=file_name: self.open_report(
                                       os.path.join(self.folder_path, path)))
                button.pack(padx=10, pady=20)

    def open_report(self, file_path):
        report = Report(file_path)
        report.mainloop()

    def get_file_name(self, file_path):
        # Get the base name of the file (with extension)
        file_name_with_extension = os.path.basename(file_path)

        # Remove the file extension
        file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

        return file_name_without_extension



class Page3(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Settings", font=LARGEFONT)
        label.grid(row=0, column=2, padx=10, pady=10)

        # button to show frame 3 with text
        # layout3
        button1 = tk.Button(self, text="Back to Main Menu",
                            command=lambda: controller.show_page(StartPage))

        # putting the button in its place by
        # using grid
        button1.grid(row=0, column=0, padx=10, pady=10)

        button2 = tk.Button(self, text="Colour Customisation")

        # putting the button in its place by
        # using grid
        button2.grid(row=1, column=2, padx=10, pady=10)

        button3 = tk.Button(self, text="Test Video Input")

        # putting the button in its place by
        # using grid
        button3.grid(row=2, column=2, padx=10, pady=10)

class Page4(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent,)
        label = tk.Label(self, text="Choose Input", font=LARGEFONT)
        label.grid(row=0, column=2, padx=10, pady=10)

        # button to show frame 3 with text
        # layout3
        button1 = tk.Button(self, text="Back to Mode Select",
                            command=lambda: controller.show_page(Page1))

        # putting the button in its place by
        # using grid
        button1.grid(row=0, column=0, padx=10, pady=10)

        button2 = tk.Button(self, text="Webcam",bg = "black",command = lambda:Diagnosis_Mode())

        # putting the button in its place by
        # using grid
        button2.grid(row=1, column=2, padx=10, pady=10)

        button3 = tk.Button(self, text="Video from Device")

        # putting the button in its place by
        # using grid
        button3.grid(row=2, column=2, padx=10, pady=10)
