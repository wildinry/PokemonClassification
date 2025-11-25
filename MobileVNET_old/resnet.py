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
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from PIL import Image

# --- HYPERPARAMETERS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# ----------------------------------------
# 1. DATA LOADING AND PREPROCESSING (Unchanged)
# ----------------------------------------
print("1. Starting Data Loading and Preprocessing...")

df = pd.read_csv("pokemon.csv")
images = []
labels = []
image_dir = "images copy"

for idx, row in df.iterrows():
    # ... (Loading image logic is unchanged)
    name = row["Name"].lower().replace(' ', '_')
    img_filename = name + ".png"        
    img_path = os.path.join(image_dir, img_filename)

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    images.append(np.array(img))
    labels.append(row["Type1"])        

X = np.array(images)
y = np.array(labels)

# Label Encoding and Split
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)
NUM_CLASSES = len(encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, shuffle=True
)

# CRITICAL STEP 1: Preprocess and convert to float32
# Note: We apply MobileNetV2 preprocessing to the whole dataset now
X_train_proc = preprocess_input(X_train.astype(np.float32))
X_test_proc = preprocess_input(X_test.astype(np.float32))

print(f"Data ready. Train samples: {len(X_train)}, Classes: {NUM_CLASSES}")


# ----------------------------------------
# 2. DATA AUGMENTATION & GENERATOR
# ----------------------------------------
# CRITICAL FIX: Use the stable ImageDataGenerator for augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the augmentation parameters (only simple transforms now)
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
# Note: Since MobileNetV2 preprocessing is already done,
# we do not need the 'rescale' or 'preprocessing_function' arguments here.

# Create the generator from the pre-processed NumPy array
train_generator = train_datagen.flow(
    X_train_proc, y_train,
    batch_size=BATCH_SIZE
)

# ----------------------------------------
# 3. MODEL BUILD (MobileNetV2, unchanged)
# ----------------------------------------
print("\n2. Building MobileNetV2 Transfer Learning Model...")

base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=IMAGE_SIZE + (3,)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(256, activation='relu')(x) 
x = Dropout(0.5)(x) 
predictions = Dense(NUM_CLASSES, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------------------
# 4. TRAINING: Use the Generator
# ----------------------------------------
print("\n3. Starting Training (WATCH FOR PROGRESS BAR NOW!)...")

# Use steps_per_epoch because we're using a generator
model.fit(
    train_generator,
    steps_per_epoch=len(X_train_proc) // BATCH_SIZE,
    validation_data=(X_test_proc, y_test),
    epochs=EPOCHS
)


# ----------------------------------------
# 5. EVALUATION AND SAVING
# ----------------------------------------
print("\n4. Evaluation and Saving...")

# Use the pre-processed test data
test_loss, test_accuracy = model.evaluate(X_test_proc, y_test, verbose=1)

print("\n--- FINAL RESULTS ---")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("---------------------")

model.save("pokemon_mobilenet_generator_model.h5")
np.save("label_classes.npy", encoder.classes_)

print("Model and label classes saved!")
