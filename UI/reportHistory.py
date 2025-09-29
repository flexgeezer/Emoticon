import tkinter as tk

class ReportHistory(tk.Frame):
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

class Report(tk.Tk):
    def __init__(self, file_path):
        tk.Tk.__init__(self)
        self.geometry("1000x2000")
        global emoji_dict
        self.grid_rowconfigure(0, weight=1)  # Make row expandable
        self.grid_columnconfigure(0, weight=1)
        self.array = self.readarray(file_path)
        self.comments = self.readcomment(file_path)
        self.emojipath = self.emojiDisplay(self.array)

        label = tk.Label(self, text= self.get_file_name(file_path), font=LARGEFONT)
        label.grid(row=0, column=0, columnspan=self.winfo_width(), padx=0, pady=0)


        label = tk.Label(self, text="Comments:")
        label.grid(row=2, column=0, padx=10, pady=10, sticky="W")

        comment_box = tk.Text(self, height=10, width=2000, font="Arial", relief=tk.GROOVE)
        comment_box.grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        comment_box.insert("1.0", self.comments)

        save_button = tk.Button(self, text="Save",
                                command=lambda: self.savedata(file_path, comment_box.get("1.0", "end-1c")))
        save_button.grid(row=2, column=2, padx=10, pady=10)

        graph_frame = tk.Frame(self)
        graph_frame.grid(row=1, column=0, sticky="nsew")  # Fill the entire grid cell

        self.plot_line_graph(graph_frame, self.array, 1)
        image = Image.open(self.emojipath)
        image = image.resize((200, 200))  # Resize image as needed
        photo = ImageTk.PhotoImage(image)

        # Create label to display image
        image_label = tk.Label(self, image=photo)
        image_label.photo = photo  # Keep reference to prevent garbage collection
        image_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")


    def plot_line_graph(self, parent, array, scale, title="Emotion Graph",
                        emotions=("Happy", "Sad", "Confused", "Angry"), x_label=" Frames",
                        y_label="Confidence Score"):
        x_values = []
        y_values = array
        for i, j in enumerate(array):
            x_values.append(i * scale)
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot(x_values, y_values, label=emotions)
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return canvas

    def emojiDisplay(self, array):
        emo_index = np.argmax(np.mean(array, axis=0))
        emoji_path = emoji_dict.get(emo_index)
        return str(emoji_path)


    def readcomment(self, file_path):
        with open(file_path, mode='r') as file:
            text = ""
            next(file)  # Skip the first line
            for line in file:
                text += line  # Concatenate each line into the text block

        return text

    def readarray(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            single_row = rows[0]
            # Split the single row into individual elements
            elements = single_row

            # Convert the elements into a 2D array
            array = [[float(element) for element in elements[i:i + 4]] for i in range(0, len(elements), 4)]
        return array

    def savedata(self, file_path, text):
        with open(file_path, 'r') as file:
            first_line = file.readline()
        with open(file_path, 'w') as file:
            file.write(first_line)  # Rewrite the first line

        with open(file_path, 'a') as file:
            file.write(text)

    def get_file_name(self,file_path):
        # Get the base name of the file (with extension)
        file_name_with_extension = os.path.basename(file_path)

        # Remove the file extension
        file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

        return file_name_without_extension

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