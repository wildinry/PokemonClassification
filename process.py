import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sys import exit
from math import isnan as isNull

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
    # to std size
    img = img.crop(crop_box)

    # print(img.size)
    # img.show()
    # exit()

    # to enlarge image
    img = img.resize((target_dim, target_dim))
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.show()
    # flipped_img.show()


    img = np.array(img)

    images.append(img)
    labels.append(row["Type1"])        
    print(type(row["Type2"]) is str)

    if type(row["Type2"]) is str:
        images.append(img)
        print("Addding second str")

print("Loaded images:", len(images))

