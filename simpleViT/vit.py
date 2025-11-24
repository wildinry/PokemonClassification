import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys

# -----------------------------
# Simple Progress Bar Function
# -----------------------------
def progress_bar(current, total, bar_length=40):
    fraction = current / total
    filled = int(fraction * bar_length)
    bar = "█" * filled + "-" * (bar_length - filled)
    percent = int(fraction * 100)
    sys.stdout.write(f"\rLoading images: |{bar}| {percent}% ({current}/{total})")
    sys.stdout.flush()

# -----------------------------
# Settings (optimized for CPU)
# -----------------------------
IMG_SIZE = (96, 96)   # small = faster
BATCH = 16
EPOCHS = 5

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("pokemon.csv")
print("Loaded CSV rows:", len(df))
print(df.head(), "\n")

# -----------------------------
# Load images with progress bar
# -----------------------------
image_dir = "images"
images = []
labels = []

total_rows = len(df)

for idx, row in df.iterrows():
    filename = row["Name"].lower() + ".png"
    path = os.path.join(image_dir, filename)

    if not os.path.exists(path):
        print(f"\nMissing image: {path}")
        continue

    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    img = np.array(img, dtype="float32") / 255.0

    images.append(img)
    labels.append(row["Type1"])

    progress_bar(idx + 1, total_rows)

print("\nImages loaded.")

X = np.array(images, dtype="float32")
y = np.array(labels)

print("Dataset:", X.shape, y.shape)

# -----------------------------
# Encode labels
# -----------------------------
encoder = LabelEncoder()
y_encoded = to_categorical(encoder.fit_transform(y))
num_classes = y_encoded.shape[1]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, shuffle=True
)

# -----------------------------
# tf.data pipeline
# -----------------------------
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(300) \
    .batch(BATCH) \
    .prefetch(1)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    .batch(BATCH) \
    .prefetch(1)

# -----------------------------
# CPU-friendly EfficientNetB0
# -----------------------------
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
    weights="imagenet"
)

# Freeze most layers
for layer in base.layers[:-30]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

# -----------------------------
# Save model + labels
# -----------------------------
model.save("pokemon_model.h5")
np.save("label_classes.npy", encoder.classes_)

print("Training complete. Model saved.")
