import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("pokemon.csv")
print("Loaded CSV rows:", len(df))
print(df.head())

# -----------------------------
# Setup image directory
# -----------------------------
image_dir = "images"

images = []
labels = []

# -----------------------------
# Load images based on Name
# -----------------------------
for idx, row in df.iterrows():
    name = row["Name"].lower()       
    img_filename = name + ".png"        
    img_path = os.path.join(image_dir, img_filename)

    if not os.path.exists(img_path):
        print("Missing image:", img_path)
        continue

    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0      

    images.append(img)
    labels.append(row["Type1"])        

print("Loaded images:", len(images))

# Convert to arrays
X = np.array(images)
y = np.array(labels)

print("Dataset shape:", X.shape, y.shape)

# -----------------------------
# Label encode types
# -----------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

print("Number of classes:", len(encoder.classes_))

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, shuffle=True
)

# -----------------------------
# Simple CNN model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# -----------------------------
# Train model
# -----------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# -----------------------------
# Evaluate model on test set
# -----------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
# Save model + label encoder
# -----------------------------
model.save("pokemon_model.h5")
np.save("label_classes.npy", encoder.classes_)

print("Model and label classes saved!")
