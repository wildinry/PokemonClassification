import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from typing import List, Tuple

IMAGE_SIZE: Tuple[int, int] = (224, 224)
RANDOM_SEED: int = 42
DATASET_PATH: str = 'dataset'
CSV_PATH: str = 'pokemon_alopez247.csv'
MODEL_FILENAME: str = 'pokemon_card_type_model.h5'
TARGET_COLUMN: str = 'Type_1'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

EPSILON_STEPS: List[float] = [0.00, 0.005, 0.01, 0.02, 0.05, 0.07, 0.10]
TEST_SAMPLE_COUNT: int = 100 # Number of images to use for the adversarial test

def preprocess_image(img):
    """Applies the same preprocessing function used during model training."""
    # MobileNetV2 preprocessing scales input images to the range [-1, 1]
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

def create_image_dataframe(csv_path: str, dataset_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Scans the dataset directory, collects all image paths, and merges with CSV data."""
    print(f"\n--- Preparing Image and Metadata DataFrame ---")
    
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
        print(f"CRITICAL ERROR: Found 0 images. Please verify paths.")
        return None, None
        
    df_images = pd.DataFrame(image_paths)
    print(f"Collected {len(df_images)} total image paths. Using the first {TEST_SAMPLE_COUNT} for test.")
    
    df_test = df_images.sample(TEST_SAMPLE_COUNT, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_test, all_possible_classes

def create_adversarial_pattern(input_image, input_label, model: Model):
    
    with tf.GradientTape() as tape:
        # Reshape to [1, H, W, 3] and watch the input tensor
        if not tf.is_tensor(input_image):
            input_image = tf.convert_to_tensor(input_image)

        input_image = input_image[None, ...]
        tape.watch(input_image)
        
        prediction = model(input_image)
        loss = tf.keras.losses.CategoricalCrossentropy()(input_label[None, ...], prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad[0]

def generate_adversarial_images(df_test: pd.DataFrame, classes: List[str], model: Model, epsilon: float):
    adversarial_images = []
    true_labels_one_hot = []
    class_indices = {name: i for i, name in enumerate(classes)}
    
    for index, row in df_test.iterrows():
        img = load_img(row['filepath'], target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        preprocessed_img_np = preprocess_image(img_array)
        
        preprocessed_img_tensor = tf.convert_to_tensor(preprocessed_img_np)

        true_label_index = class_indices[row[TARGET_COLUMN]]
        one_hot_label = tf.one_hot(true_label_index, depth=len(classes))
        
        # If epsilon is 0.00, we skip perturbation and use the original preprocessed image tensor
        if epsilon == 0.00:
            adv_img_tensor = preprocessed_img_tensor
        else:
            perturbation = create_adversarial_pattern(preprocessed_img_tensor, one_hot_label, model)
            
            # Create the adversarial image: original + epsilon * perturbation
            adv_img_tensor = preprocessed_img_tensor + epsilon * perturbation
            adv_img_tensor = tf.clip_by_value(adv_img_tensor, -1.0, 1.0) 
            
        adversarial_images.append(adv_img_tensor.numpy())
        true_labels_one_hot.append(one_hot_label.numpy())
    
    return np.array(adversarial_images), np.array(true_labels_one_hot)

def evaluate_and_plot(adv_images, true_labels_one_hot, model: Model, classes: List[str], epsilon: float):
    """Evaluates the model on adversarial images and plots a sample."""
    adv_predictions = model.predict(adv_images, verbose=0)
    predicted_classes = np.argmax(adv_predictions, axis=1)
    true_classes = np.argmax(true_labels_one_hot, axis=1)
    correct_predictions = np.sum(predicted_classes == true_classes)
    adversarial_accuracy = correct_predictions / len(adv_images)
    
    print(f"\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print(f"| Evaluation for Epsilon = {epsilon:.3f}")
    print(f"| Samples Tested: {len(adv_images)}")
    print(f"| Adversarial Accuracy: {adversarial_accuracy:.4f} (Correct: {correct_predictions})")
    if epsilon > 0.0:
        accuracy_drop = (1 - adversarial_accuracy) * 100
        print(f"| Failure Rate (Adversarially Fooled): {accuracy_drop:.2f}%")
    print(f"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    if epsilon > 0.0 and epsilon in [0.01, 0.05, 0.10]:
        plot_adversarial_samples(adv_images, predicted_classes, true_classes, classes, epsilon)

def plot_adversarial_samples(images, predicted_classes, true_classes, class_names, epsilon):
    """Displays a sample of adversarial images, highlighting successful attacks."""
    
    is_correct = (predicted_classes == true_classes)
    fooled_indices = np.where(~is_correct)[0]
    
    if len(fooled_indices) < 5:
        sample_indices = np.arange(5)
        plot_title = f"First 5 Images (Epsilon: {epsilon:.3f})"
    else:
        np.random.shuffle(fooled_indices)
        sample_indices = fooled_indices[:5]
        plot_title = f"5 Successful Adversarial Attacks (Epsilon: {epsilon:.3f})"

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle(plot_title, fontsize=14, y=1.05)
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Rescale image from [-1, 1] to [0, 1] for display
        img = (images[idx] + 1) / 2
        ax.imshow(img) 
        
        pred_name = class_names[predicted_classes[idx]]
        true_name = class_names[true_classes[idx]]
        color = 'red' if predicted_classes[idx] != true_classes[idx] else 'green'

        title_text = f"Pred: {pred_name}\nTrue: {true_name}"
        ax.set_title(title_text, color=color, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def run_adversarial_test():
    """Main function to run the adversarial robustness test."""
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Model Filename: {MODEL_FILENAME}")

    df_test, classes = create_image_dataframe(CSV_PATH, DATASET_PATH)
    if df_test is None or not classes:
        return
    if not os.path.exists(MODEL_FILENAME):
        print(f"\nERROR: Saved model not found at '{MODEL_FILENAME}'. Please ensure your training script has run successfully.")
        return

    try:
        model = tf.keras.models.load_model(MODEL_FILENAME)
        print(f"\n✅ Successfully loaded model from {MODEL_FILENAME}.")
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        return

    print(f"\n--- Starting Adversarial Robustness Test on {len(df_test)} Samples ---")
    
    base_images, true_labels_one_hot = generate_adversarial_images(
        df_test.copy(), classes, model, epsilon=0.00
    )

    for epsilon in EPSILON_STEPS:
        if epsilon == 0.00:
            adv_images = base_images
            true_labels = true_labels_one_hot
        else:
            adv_images, true_labels = generate_adversarial_images(
                df_test.copy(), classes, model, epsilon=epsilon
            )
            
        evaluate_and_plot(adv_images, true_labels, model, classes, epsilon)
        
    print("\nAdversarial testing complete.")

if __name__ == '__main__':
    run_adversarial_test()
