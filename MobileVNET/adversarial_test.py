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
TEST_SAMPLE_COUNT: int = 100

def preprocess_image(img):
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

def create_image_dataframe(csv_path: str, dataset_path: str) -> Tuple[pd.DataFrame, List[str]]:
    df_meta = pd.read_csv(csv_path)[['Name', TARGET_COLUMN]]
    name_to_type = df_meta.set_index('Name')[TARGET_COLUMN].to_dict()
    all_possible_classes = sorted(df_meta[TARGET_COLUMN].unique())

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

    df_images = pd.DataFrame(image_paths)
    df_test = df_images.sample(TEST_SAMPLE_COUNT, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_test, all_possible_classes

def create_adversarial_pattern(input_image, input_label, model: Model):
    with tf.GradientTape() as tape:
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
    
    for _, row in df_test.iterrows():
        img = load_img(row['filepath'], target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        preprocessed = preprocess_image(img_array)
        pre_tensor = tf.convert_to_tensor(preprocessed)

        true_label_index = class_indices[row[TARGET_COLUMN]]
        one_hot = tf.one_hot(true_label_index, depth=len(classes))

        if epsilon == 0.00:
            adv_img_tensor = pre_tensor
        else:
            perturbation = create_adversarial_pattern(pre_tensor, one_hot, model)
            ## main alteration takes pretensor (image converted to tf tensor) epsilon is
            ## the "strength" of adversarial attack, and pertubation is a "map" to show 
            ## which pixels to alter
            adv_img_tensor = pre_tensor + epsilon * perturbation
            adv_img_tensor = tf.clip_by_value(adv_img_tensor, -1.0, 1.0) 

        adversarial_images.append(adv_img_tensor.numpy())
        true_labels_one_hot.append(one_hot.numpy())
    
    return np.array(adversarial_images), np.array(true_labels_one_hot)

def evaluate_and_plot(adv_images, true_labels_one_hot, model: Model, classes: List[str], epsilon: float):
    adv_predictions = model.predict(adv_images, verbose=0)
    predicted = np.argmax(adv_predictions, axis=1)
    truth = np.argmax(true_labels_one_hot, axis=1)
    correct = np.sum(predicted == truth)
    accuracy = correct / len(adv_images)
    print(f"Epsilon = {epsilon:.3f} | Accuracy = {accuracy:.4f}")
    return accuracy

def plot_accuracy_vs_epsilon(epsilons, accuracies):
    plt.figure(figsize=(8,5))
    plt.plot(epsilons, accuracies, marker='o')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("Model Robustness vs Epsilon")
    plt.grid(True)
    plt.show()

def run_adversarial_test():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df_test, classes = create_image_dataframe(CSV_PATH, DATASET_PATH)
    if df_test is None:
        return

    model = tf.keras.models.load_model(MODEL_FILENAME)

    base_images, true_labels = generate_adversarial_images(
        df_test.copy(), classes, model, epsilon=0.00
    )

    recorded_accuracies = []

    for epsilon in EPSILON_STEPS:
        if epsilon == 0.00:
            adv_images = base_images
            labels = true_labels
        else:
            adv_images, labels = generate_adversarial_images(
                df_test.copy(), classes, model, epsilon
            )

        acc = evaluate_and_plot(adv_images, labels, model, classes, epsilon)
        recorded_accuracies.append(acc)

    plot_accuracy_vs_epsilon(EPSILON_STEPS, recorded_accuracies)

if __name__ == '__main__':
    run_adversarial_test()
