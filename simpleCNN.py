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

edge_crop = 20 # pixels
target_dim = 120
input_dim = 120
crop_box = (edge_crop, edge_crop, target_dim-edge_crop, target_dim-edge_crop)
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

    img = Image.open(img_path).convert("RGBA")
    # img = img.crop(crop_box)

    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(img) 
    flipped_img = np.array(flipped_img)

    images.append(img)
    images.append(flipped_img)
    labels.append(row["Type1"])        
    labels.append(row["Type1"])        

    if type(row["Type2"]) is str:
        images.append(img)
        images.append(flipped_img)
        labels.append(row["Type2"])        
        labels.append(row["Type2"])        
        print("Adding second type")


print("Loaded images:", len(images))

# Convert to arrays
X = np.array(images)
y = np.array(labels)

# def augment_image_with_flip(image, label):
#     # Apply horizontal flip
#     flipped_image = tf.image.flip_left_right(image)
#     return flipped_image, label

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
    X, y_cat, test_size=0.1, random_state=67, shuffle=True
)

# -----------------------------
# Simple CNN model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(input_dim,input_dim,4)),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(256, (3,3), activation="relu"),
    tf.keras.layers.Conv2D(256, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # tf.keras.layers.Conv2D(512, (3,3), activation="relu"),
    # tf.keras.layers.Conv2D(512, (3,3), activation="relu"),
    # tf.keras.layers.Conv2D(512, (3,3), activation="relu"),
    # tf.keras.layers.MaxPooling2D(),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(2048, activation="relu"),
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
history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

import matplotlib.pyplot as plt
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy'] # If you used validation data
epochs = range(1, len(train_accuracy) + 1)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy') # If you used validation data
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

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
