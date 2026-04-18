import numpy as np
import cv2
from config import emoji_dict
from Modes.helpers import load_model, preprocess, standby

np.set_printoptions(suppress=True)


class Live_Mode():
    def __init__(self):
        self.colors = ['#00FF00', '#0000FF', '#FFFF00', '#FF0000']
        standby('Press "Enter" to start')

        model, class_names = load_model()
        self.camera = cv2.VideoCapture(0)

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print('Video isnt working')
                break

            prediction_np = model(preprocess(frame)).numpy()
            index = np.argmax(prediction_np)

            print(class_names[index])
            print(prediction_np[0][index])
            print(prediction_np)

            self.LiveMode(index)

            if cv2.waitKey(1) == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()

    def tint_frame(self, hex_color, opacity=0.3):
        ret, image = self.camera.read()
        color_rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))[::-1]
        color_layer = np.full_like(image, color_rgb, dtype=np.uint8)
        return cv2.addWeighted(image, 1 - opacity, color_layer, opacity, 0)

    def emojiOverlay(self, index, frame):
        emoji = cv2.imread(emoji_dict.get(index))
        emoji = cv2.resize(emoji, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(frame, 1, emoji, 0.5, 0)

    def LiveMode(self, index):
        tinted_frame = self.tint_frame(self.colors[index])
        cv2.imshow("Emoticon", self.emojiOverlay(index, tinted_frame))
