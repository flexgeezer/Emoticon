import csv
import os
import time
import numpy as np
import cv2
from tkinter import simpledialog
from config import EMOTIVE_DATA_DIR
from Modes.helpers import load_model, preprocess, standby


class Diagnosis_Mode():
    def __init__(self, video_source=0):
        self.reportarray = []
        self.timestamps = []
        self.emoticon_file = ''

        model, class_names = load_model()

        username = self.nameprompt()
        if not username:
            return
        self.emoticon_file = os.path.join(EMOTIVE_DATA_DIR, f'{username}.csv')

        if video_source == 0:
            standby('Press "Enter" to Diagnose')

        start_time = time.time()
        camera = cv2.VideoCapture(video_source)

        while camera.isOpened():
            ret, image = camera.read()
            if not ret:
                break

            cv2.imshow("Emoticon", image)

            prediction_np = model(preprocess(image)).numpy()
            index = np.argmax(prediction_np)

            print("Class:", class_names[index][2:], end="")
            print("Confidence Score:", str(np.round(prediction_np[0][index] * 100))[:-2], "%")

            elapsed = time.time() - start_time
            self.reportarray.append(prediction_np.flatten().tolist())
            self.timestamps.append(round(elapsed, 3))

            if cv2.waitKey(1) == 27 or elapsed >= 60:
                break

        camera.release()
        cv2.destroyAllWindows()
        self.createReport(self.reportarray, self.timestamps)

    def nameprompt(self):
        return simpledialog.askstring("Name", "Please enter your name:")

    def createReport(self, array, timestamps):
        if not self.emoticon_file or not array:
            return
        with open(self.emoticon_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['frame', 'time', 'happy', 'sad', 'confused', 'angry'])
            for i, (row, t) in enumerate(zip(array, timestamps)):
                writer.writerow([i, t] + [round(v, 6) for v in row])
            file.write('\n')
