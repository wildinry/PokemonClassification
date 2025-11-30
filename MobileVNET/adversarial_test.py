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

EPSILON_STEPS: List[float] = [
    0.00, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95, 1.00
]
TEST_SAMPLE_COUNT: int = 100

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def preprocess_image(img):
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

def create_image_dataframe(csv_path: str, dataset_path: str):
    df_meta = pd.read_csv(csv_path)[['Name', TARGET_COLUMN]]
    name_to_type = df_meta.set_index('Name')[TARGET_COLUMN].to_dict()
    all_classes = sorted(df_meta[TARGET_COLUMN].unique())
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        if root == dataset_path:
            continue
        pokemon_name = os.path.basename(root)
        if pokemon_name in name_to_type:
            t = name_to_type[pokemon_name]
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    image_paths.append({
                        "filepath": os.path.join(root, file),
                        "Name": pokemon_name,
                        TARGET_COLUMN: t
                    })
    df_images = pd.DataFrame(image_paths)
    df_test = df_images.sample(TEST_SAMPLE_COUNT, random_state=RANDOM_SEED).reset_index(drop=True)
    return df_test, all_classes

def create_adversarial_pattern(input_image, input_label, model: Model):
    with tf.GradientTape() as tape:
        if not tf.is_tensor(input_image):
            input_image = tf.convert_to_tensor(input_image)
        input_image = input_image[None, ...]
        tape.watch(input_image)
        pred = model(input_image)
        loss = tf.keras.losses.CategoricalCrossentropy()(input_label[None, ...], pred)
    grad = tape.gradient(loss, input_image)
    return tf.sign(grad)[0]

def generate_adversarial_images(df_test, classes, model, epsilon):
    adv = []
    labels = []
    idx = {c: i for i, c in enumerate(classes)}
    for _, row in df_test.iterrows():
        img = load_img(row['filepath'], target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        p = preprocess_image(img_array)
        tensor = tf.convert_to_tensor(p)
        label_index = idx[row[TARGET_COLUMN]]
        onehot = tf.one_hot(label_index, depth=len(classes))
        if epsilon == 0:
            out = tensor
        else:
            pert = create_adversarial_pattern(tensor, onehot, model)
            out = tensor + epsilon * pert
            out = tf.clip_by_value(out, -1.0, 1.0)
        adv.append(out.numpy())
        labels.append(onehot.numpy())
    return np.array(adv), np.array(labels)

def save_grid(images, epsilon):
    imgs = (images + 1) / 2
    plt.figure(figsize=(15, 15))
    cols = 10
    rows = 10
    for i in range(100):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i])
        plt.axis("off")
    path = os.path.join(RESULTS_DIR, f"grid_eps_{epsilon:.2f}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def run_adversarial_test():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    df_test, classes = create_image_dataframe(CSV_PATH, DATASET_PATH)
    model = tf.keras.models.load_model(MODEL_FILENAME)
    base_imgs, base_lbls = generate_adversarial_images(df_test.copy(), classes, model, epsilon=0.00)
    eps_list = []
    acc_list = []

    for eps in EPSILON_STEPS:
        if eps == 0:
            imgs = base_imgs
            labels = base_lbls
        else:
            imgs, labels = generate_adversarial_images(df_test.copy(), classes, model, eps)
        preds = model.predict(imgs, verbose=0)
        pred = np.argmax(preds, axis=1)
        truth = np.argmax(labels, axis=1)
        acc = np.sum(pred == truth) / len(imgs)
        eps_list.append(eps)
        acc_list.append(acc)
        save_grid(imgs, eps)

    plt.figure(figsize=(8, 5))
    plt.plot(eps_list, acc_list, marker='o')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Epsilon")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=300)
    plt.close()

if __name__ == '__main__':
    run_adversarial_test()
