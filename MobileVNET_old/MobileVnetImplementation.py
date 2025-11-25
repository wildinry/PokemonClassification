import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ============================================================
# Load CSV
# ============================================================
df = pd.read_csv("pokemon.csv")
print("Loaded CSV rows:", len(df))

image_dir = "images"

images = []
labels = []

# ============================================================
# Load and preprocess images
# ============================================================
for idx, row in df.iterrows():
    name = row["Name"].lower()
    filename = f"{name}.png"
    path = os.path.join(image_dir, filename)

    if not os.path.exists(path):
        print(f"Missing image: {filename}")
        continue

    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
        img = np.array(img) / 255.0  
        images.append(img)
        labels.append(row["Type1"])
    except:
        print(f"Error loading: {filename}")

images = np.array(images)
labels = np.array(labels)

print("Images loaded:", len(images))
print("Dataset shape:", images.shape)

# ============================================================
# Encode labels
# ============================================================
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)

y_cat = tf.keras.utils.to_categorical(y, num_classes=num_classes)

print("Class count:", num_classes)
print("Classes:", encoder.classes_)

# ============================================================
# Train/val/test split
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    images, y_cat, test_size=0.30, random_state=42, shuffle=True
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# ============================================================
# Data Augmentation
# ============================================================
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.1),
])

# ============================================================
# Transfer Learning Model (MobileNetV2)
# ============================================================
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  # Freeze base layers

model = tf.keras.Sequential([
    data_aug,
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# Callbacks
# ============================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "pokemon_model_best.h5",
        save_best_only=True,
        monitor="val_loss"
    )
]

# ============================================================
# Train
# ============================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# ============================================================
# Evaluate on Test Set
# ============================================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# ============================================================
# Save Final Model and Label Classes
# ============================================================
model.save("pokemon_model_final.h5")
np.save("label_classes.npy", encoder.classes_)
print("Model and label classes saved!")
