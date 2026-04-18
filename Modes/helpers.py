import time
import numpy as np
import tensorflow as tf
import cv2
from config import MODEL_PATH, LABELS_PATH


def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        class_names = f.readlines()
    return model, class_names


def preprocess(frame):
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
    frame = (frame / 127.5) - 1
    return tf.convert_to_tensor(frame)


def standby(message):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, textframe = cap.read()
        if not ret:
            continue
        (h, w) = textframe.shape[:2]
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        x = (w - text_size[0]) // 2
        y = (h + text_size[1]) // 2
        cv2.putText(textframe, message, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Prompt', textframe)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            time.sleep(1)
            cap.release()
            cv2.destroyAllWindows()
            break
        if key == 3:  # Ctrl+C
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt
