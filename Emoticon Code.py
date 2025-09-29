import tensorflow as tf
import cv2  # Install opencv-python
import tkinter as tk
import os
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import sys
from tkinter import simpledialog

LARGEFONT = ("Verdana", 35)

# Change all redirectories

emoji_dict = {
    0: "/Users/Admin/PycharmProjects/Emoticon/Emojis/Emoji0.png",
    1: "/Users/Admin/PycharmProjects/Emoticon/Emojis/Emoji1.png",
    2: "/Users/Admin/PycharmProjects/Emoticon/Emojis/Emoji2.png",
    3: "/Users/Admin/PycharmProjects/Emoticon/Emojis/Emoji3.png"
}


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
        for F in (StartPage, Page1, Page2,Page3,Page4):
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

class Live_Mode():
    def __init__(self):
        self.colors = ['#00FF00', '#0000FF', '#FFFF00', '#FF0000']
        self.Standby_L()
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = tf.saved_model.load("model.savedmodel") # Change placeholder to tensorflow emotional vision model 

        # Load the labels
        class_names = open("labels.txt", "r").readlines()

        # CAMERA can be 0 or 1 based on default camera of your computer
        self.camera = cv2.VideoCapture(0)

        while True:
            # Grab the webcamera's image.
            ret, frame = self.camera.read()
            # Show the image in a window

            if ret == False:
                print('Video isnt working')
                break

            # Resize the raw image into (224-height,224-width) pixels
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

            # Make the image a numpy array and reshape it to the models input shape.
            frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalise the image array
            frame = (frame / 127.5) - 1

            image_tensor = tf.convert_to_tensor(frame)

            # Predict the model
            predictions = model(image_tensor)

            # Convert predictions to numpy array

            prediction_np = predictions.numpy()

            # Get the predicted class index
            index = np.argmax(prediction_np)

            # Get the predicted class name and confidence score
            class_name = class_names[index]
            confidence_score = prediction_np[0][index]
            self.LiveMode(index)

            # Print prediction and confidence score
            print(class_name)
            print(confidence_score)
            print(prediction_np)

            # Listen to the keyboard for presses
            keyboard_input = cv2.waitKey(1)

            # 27 is the ASCII for the esc key on your keyboard
            if keyboard_input == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()

    def tint_frame(self,hex_color, opacity=0.3):
        # Convert hexadecimal color code to BGR format
        ret, image = self.camera.read()
        color_rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))[::-1]

        # Create a colored layer with the specified color
        color_layer = np.full_like(image, color_rgb, dtype=np.uint8)

        # Blend the colored layer with the original frame using alpha blending
        tinted_frame = cv2.addWeighted(image, 1 - opacity, color_layer, opacity, 0)

        return tinted_frame


    def emojiOverlay(self,index,frame):
        emoji = emoji_dict.get(index)
        emoji = cv2.imread(emoji)
        emoji = cv2.resize(emoji, (frame.shape[1], frame.shape[0]))
        frame =cv2.addWeighted(frame, 1, emoji, 0.5, 0)
        return frame
    def LiveMode(self,index):
        tinted_frame = self.tint_frame(self.colors[index])
        emojiframe = self.emojiOverlay(index,tinted_frame)
        cv2.imshow("Emoticon",emojiframe)
    def Standby_L(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, textframe = cap.read()
            cv2.putText(textframe, 'Press "Enter" to start', (500, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Prompt', textframe)  # Display the frame
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Check if "Enter" key is pressed
                time.sleep(1)
                cap.release()
                cv2.destroyAllWindows()

class Diagnosis_Mode():
    def __init__(self):

        self.reportarray = []
        self.emoticon_file = ''
        global emoji_dict

        # Load the model
        model = tf.saved_model.load('model.savedmodel')# Change placeholder to tensorflow emotional vision model 
        # Load the labels
        class_names = open("/Users/Admin/PycharmProjects/Emoticon/labels.txt", "r").readlines()

        username = self.nameprompt()
        self.emoticon_file = f'/Users/Admin/PycharmProjects/Emoticon/Emotive Data/{username}.csv'

        self.Standby_D()
        start_time = time.time()
        camera = cv2.VideoCapture(0)
        while camera.isOpened():
            # Grab the webcamera's image.
            ret, image = camera.read()
            # Show the image in a window
            cv2.imshow("Emoticon", image)
            # Resize the raw image into (224-height,224-width) pixels
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

            # Make the image a numpy array and reshape it to the models input shape.
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalise the image array
            image = (image / 127.5) - 1

            image_tensor = tf.convert_to_tensor(image)

            # Predict the model
            predictions = model(image_tensor)

            # Convert predictions to numpy array
            prediction_np = predictions.numpy()

            # Get the predicted class index
            index = np.argmax(prediction_np)

            # Get the predicted class name and confidence score
            class_name = class_names[index]
            confidence_score = prediction_np[0][index]

            # Print prediction and confidence score
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
            print(prediction_np)

            prediction_np = np.array(prediction_np).flatten()

            prediction_np = prediction_np.tolist()
            self.reportarray.append(prediction_np)

            # Listen to the keyboard for presses
            keyboard_input = cv2.waitKey(1)

            # 27 is the ASCII for the esc key on your keyboard
            if keyboard_input == 27 or time.time() - start_time >= 60:
                camera.release()
                cv2.destroyAllWindows()


        self.createReport(self.reportarray)
        app = Report(self.emoticon_file)
        app.mainloop()

    def nameprompt(self):
        name = simpledialog.askstring("Name", "Please enter your name:")
        return name

    def Standby_D(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, textframe = cap.read()
            cv2.putText(textframe, 'Press "Enter" to Diagnose', (500, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Prompt', textframe)  # Display the frame
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Check if "Enter" key is pressed
                time.sleep(1)
                cap.release()
                cv2.destroyAllWindows()

    def createReport(self, array):
        emoarray = []
        for row in array:
            for x in row:
                emoarray.append(x)
        string = str(emoarray)
        string = string[1:-1]
        # Writing data into the CSV file
        with open(self.emoticon_file, 'w', newline='') as file:
            file.write(string)
            file.write(' \n')

    def get_path(self):
        # Your code to return the file path
        return self.emoticon_file
# Driver Code (Keeping windows open)
app =EmoticonApp()
app.mainloop()
