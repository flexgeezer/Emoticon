import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load dataset (MNIST digits 0–9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (batch, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train for a few epochs (quick)
model.fit(x_train, y_train, epochs=2, validation_split=0.1)

# Evaluate on test set
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {acc:.3f}")

# Save model in SavedModel format
model.save("model.savedmodel")

# Save labels
with open("labels.txt", "w") as f:
    for i in range(10):
        f.write(str(i) + "\n")
