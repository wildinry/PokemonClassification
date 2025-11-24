import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom
from PIL import Image

# --- HYPERPARAMETERS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 10  # Train head quickly
EPOCHS_PHASE_2 = 15  # Fine-tune gently
LEARNING_RATE_HEAD = 1e-3 # Standard Adam for the head
LEARNING_RATE_FINE_TUNE = 1e-5 # Very low rate for fine-tuning

# ----------------------------------------
# 1. DATA LOADING AND PREPROCESSING (CRITICAL CHANGE)
# ----------------------------------------
print("1. Starting Data Loading and Preprocessing...")

df = pd.read_csv("pokemon.csv")
images = []
labels = []
image_dir = "images copy"

# Load images (no normalization here yet)
for idx, row in df.iterrows():
    name = row["Name"].lower().replace(' ', '_')
    img_filename = name + ".png"        
    img_path = os.path.join(image_dir, img_filename)

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    images.append(np.array(img))
    labels.append(row["Type1"])        

X = np.array(images).astype(np.float32) # Ensure float32 for GPU stability
y = np.array(labels)

# Label Encoding and Split
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)
NUM_CLASSES = len(encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, shuffle=True
)

print(f"Data ready. Train samples: {len(X_train)}, Classes: {NUM_CLASSES}")


# ----------------------------------------
# 2. DATA AUGMENTATION & PIPELINE (Optimized for Colab GPU)
# ----------------------------------------

# Define the Keras augmentation layers
data_augmenter = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.15),
], name='data_augmentation')

def process_data(image, label):
    """Applies preprocessing needed by MobileNetV2."""
    image = preprocess_input(image) 
    return image, label

def augment_and_process(image, label):
    """Applies augmentation and then MobileNetV2 preprocessing."""
    image = data_augmenter(image, training=True)
    return process_data(image, label)

# Build the TF Data Pipelines
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)) \
    .map(augment_and_process, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------
# 3. MODEL BUILD (MobileNetV2)
# ----------------------------------------
print("\n2. Building MobileNetV2 Transfer Learning Model...")

base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=IMAGE_SIZE + (3,)
)

# ----------------------------------------
# 4. PHASE 1: Train Only the New Classification Head
# ----------------------------------------
print("\n3. Starting PHASE 1: Training Classification Head...")

# Freeze the base layers
for layer in base_model.layers:
    layer.trainable = False

# Attach the new head (must be defined AFTER freezing for model construction)
x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(256, activation='relu')(x) 
x = Dropout(0.5)(x) 
predictions = Dense(NUM_CLASSES, activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS_PHASE_1
)


# ----------------------------------------
# 5. PHASE 2: Fine-Tuning the Base Model
# ----------------------------------------
print("\n4. Starting PHASE 2: Fine-Tuning Base Model...")

# Unfreeze the last ~50 layers of MobileNetV2 (Blocks 13 and above)
for layer in model.layers:
    if layer.name.startswith('block_13') or layer.name.startswith('block_14') or \
       layer.name.startswith('block_15') or layer.name.startswith('block_16'):
        layer.trainable = True
    else:
        layer.trainable = False # Keep early layers frozen

# Re-compile with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Continue training for Phase 2
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS_PHASE_2
)


# ----------------------------------------
# 6. EVALUATION AND SAVING
# ----------------------------------------
print("\n5. Evaluation and Saving...")

test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

print("\n--- FINAL RESULTS ---")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("---------------------")

model.save("pokemon_mobilenet_final_model.h5")
np.save("label_classes.npy", encoder.classes_)

print("Model and label classes saved!")
