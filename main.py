import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import List, Tuple

# --- Configuration Constants ---
IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
RANDOM_SEED: int = 42
DATASET_PATH: str = 'dataset'
CSV_PATH: str = 'pokemon_alopez247.csv'
MODEL_FILENAME: str = 'pokemon_card_type_model.h5'
PLOT_FILENAME: str = 'training_history.png'
TARGET_COLUMN: str = 'Type_1'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def setup_environment():
    """Sets up the environment for reproducible results and checks TF version."""
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Image Extensions to check: {IMAGE_EXTENSIONS}")

# ----------------------------------------------------------------------
# 1. Data Preparation
# ----------------------------------------------------------------------

def create_image_dataframe(csv_path: str, dataset_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Scans the dataset directory, collects all image paths, and merges with CSV data."""
    print(f"\n--- 1. Preparing Image and Metadata DataFrame ---")
    
    try:
        df_meta = pd.read_csv(csv_path)[['Name', TARGET_COLUMN]]
        name_to_type = df_meta.set_index('Name')[TARGET_COLUMN].to_dict()
        all_possible_classes = sorted(df_meta[TARGET_COLUMN].unique())
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Cannot proceed.")
        return None, None

    image_paths = []
    
    for root, _, files in os.walk(dataset_path):
        if root == dataset_path:
            continue
            
        pokemon_name = os.path.basename(root)
        
        if pokemon_name in name_to_type:
            pokemon_type = name_to_type[pokemon_name]
            
            for file_name in files:
                if file_name.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, file_name)
                    image_paths.append({
                        'filepath': full_path,
                        'Name': pokemon_name,
                        TARGET_COLUMN: pokemon_type
                    })
    
    if not image_paths:
        print(f"CRITICAL ERROR: Found 0 images with extensions {IMAGE_EXTENSIONS} in subfolders of '{dataset_path}'.")
        print("Please verify image extensions and placement.")
        return None, None
        
    df_images = pd.DataFrame(image_paths)
    
    print(f"Successfully collected {len(df_images)} total image paths.")
    print(f"Found {len(df_images[TARGET_COLUMN].unique())} unique classes with images.")
    
    return df_images, all_possible_classes

# ----------------------------------------------------------------------
# 2. Image Data Loading and Preprocessing
# ----------------------------------------------------------------------

def create_generators(df_images: pd.DataFrame, classes: List[str]):
    """Splits the data and creates Data Generators using flow_from_dataframe."""
    print("\n--- 2. Creating Image Data Generators (flow_from_dataframe) ---")

    df_train = df_images.sample(frac=0.8, random_state=RANDOM_SEED)
    df_val = df_images.drop(df_train.index)
    
    def preprocess_input(img):
        return tf.keras.applications.mobilenet_v2.preprocess_input(img)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train, x_col='filepath', y_col=TARGET_COLUMN, target_size=IMAGE_SIZE, 
        classes=classes, class_mode='categorical', batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_SEED
    )

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val, x_col='filepath', y_col=TARGET_COLUMN, target_size=IMAGE_SIZE, 
        classes=classes, class_mode='categorical', batch_size=BATCH_SIZE, shuffle=False, seed=RANDOM_SEED
    )
    
    print(f"Training samples: {train_generator.samples}, Validation samples: {validation_generator.samples}")
    print(f"Number of classes: {len(classes)}")
    
    return train_generator, validation_generator

# ----------------------------------------------------------------------
# 3. Model Definition and Training (FIXED fine-tuning index)
# ----------------------------------------------------------------------

def build_transfer_learning_model(num_classes: int) -> Model:
    """Builds a Transfer Learning model using pre-trained MobileNetV2."""
    print("\n--- 3. Building Transfer Learning Model (MobileNetV2) ---")
    base_model = MobileNetV2(
        weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    print("Model Architecture Summary:")
    model.summary()
    return model

def train_model(model: Model, train_gen: tf.keras.utils.Sequence, val_gen: tf.keras.utils.Sequence, model_path: str):
    """
    Trains and fine-tunes the model in two stages and returns the training history.
    """
    print("\n--- 4. Training the Model ---")
    
    # Stage 1: Train classification head
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint_stage1 = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop_stage1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("\n--- Stage 1: Training Classification Head ---")
    
    steps_per_epoch_train = train_gen.samples // train_gen.batch_size
    steps_per_epoch_val = val_gen.samples // val_gen.batch_size
    
    history1 = model.fit(
        train_gen, epochs=15, validation_data=val_gen, callbacks=[early_stop_stage1, checkpoint_stage1],
        steps_per_epoch=steps_per_epoch_train, validation_steps=steps_per_epoch_val
    )

    # Stage 2: Fine-Tuning
    print("\n--- Stage 2: Fine-Tuning Top Layers of MobileNetV2 ---")
    
    # Reload the best model from Stage 1 to ensure we start fine-tuning from the highest val_accuracy
    model = tf.keras.models.load_model(model_path)
    model.trainable = True
    
    # FIX APPLIED: Use index [1] to access the MobileNetV2 base model
    # Freeze the first 100 layers of the base model
    for layer in model.layers[1].layers[:100]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint_stage2 = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop_stage2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history2 = model.fit(
        train_gen, epochs=30, initial_epoch=history1.epoch[-1] if history1.epoch else 15,
        validation_data=val_gen, callbacks=[early_stop_stage2, checkpoint_stage2],
        steps_per_epoch=steps_per_epoch_train, validation_steps=steps_per_epoch_val
    )
    
    # Combine history objects
    history = {}
    for key in history1.history.keys():
        history[key] = history1.history[key] + history2.history[key]
        
    print(f"\nModel training complete. Best model saved to: {model_path}")
    return history

# ----------------------------------------------------------------------
# 4. Evaluation and Plotting (Unchanged)
# ----------------------------------------------------------------------

def evaluate_model(model_path: str, val_gen: tf.keras.utils.Sequence):
    """Loads the best model and runs a final accuracy and loss test."""
    print("\n--- 5. Final Model Evaluation ---")
    
    if not os.path.exists(model_path):
        print(f"Error: Best model file not found at {model_path}. Skipping evaluation.")
        return

    model = tf.keras.models.load_model(model_path)
    
    # Evaluate the model on the validation data
    loss, accuracy = model.evaluate(val_gen, verbose=1)

    print(f"\nFinal Validation Loss: {loss:.4f}")
    print(f"Final Validation Accuracy: {accuracy:.4f}")

def plot_training_history(history: dict, plot_path: str):
    """Plots the training and validation loss and accuracy using Matplotlib."""
    print(f"\n--- 6. Generating Training History Plot ---")
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")

# ----------------------------------------------------------------------
# Main Execution Flow (Unchanged)
# ----------------------------------------------------------------------

def main():
    """Main function to run the entire ML pipeline."""
    setup_environment()

    # 1. Prepare DataFrames (Collects all image paths and maps them to types)
    df_images, classes = create_image_dataframe(CSV_PATH, DATASET_PATH)
    if df_images is None or not classes:
        return

    # 2. Create Data Generators
    train_generator, validation_generator = create_generators(df_images, classes)
    
    if train_generator.samples == 0:
        print("ERROR: Training generator has 0 samples. Cannot proceed with training.")
        return

    # 3. Build Model
    model = build_transfer_learning_model(len(classes))
    
    # 4. Train Model
    history = train_model(model, train_generator, validation_generator, MODEL_FILENAME)
    
    # 5. Evaluate Model
    evaluate_model(MODEL_FILENAME, validation_generator)

    # 6. Plot History
    plot_training_history(history, PLOT_FILENAME)


if __name__ == '__main__':
    main()