import tensorflow as tf
from tensorflow.keras import layers, models

base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.keras")
model.save(save_path)
print(f"Saved to {save_path}")
