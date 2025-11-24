import pandas as pd
from PIL import Image
import io
import sys
from sys import exit
import numpy 


if len(sys.argv) > 1 and sys.argv[1] == "download":
    # Login using e.g. `huggingface-cli login` to access this dataset
    splits = {
        'train': 'train.parquet', 
        'validation': 'validation.parquet', 
        'test': 'test.parquet'
    }
    df = pd.read_parquet("hf://datasets/JJMack/pokemon-classification-gen1-9/" + splits["train"])
    df.drop(columns=["file_name", "label", "generation","Attack", "HP", "Defense", "Sp.Attack", "Sp.Defense", "Speed"], inplace=True)
    indices_to_drop = df[df['shiny'] == 'yes'].index
    df.drop(indices_to_drop, inplace=True)

    df.to_pickle("pokemon3.pkl")
else:
    df = pd.read_pickle("pokemon3.pkl")



print(df.head())
"""
data.py [download]

A separate file that will handle processing of data for us. 
Exports X_train, Y_train
        X_val,  Y_val
        X_test, Y_test
"""


label_set = ['Grass', 'Poison', 'Fire', 'Flying', 'Water', 'Bug', 'Normal', 'Electric', 'Ground', 'Fairy', 'Fighting', 'Psychic', 'Rock', 'Steel', 'Ice', 'Ghost', 'Dragon', 'Dark']

target_dim = 256
def process_png(bytes) -> Image.Image:
    img = Image.open(io.BytesIO(bytes))
    if img.size != (target_dim, target_dim):
        return numpy.nan
    return img

# for the multiclass classification encoding
def get_encoding(type1, type2):
    lbls = [type1]
    if type(type2) is str:
        lbls.append(type2)
    encoding = numpy.zeros(18)
    for idx, lbl in enumerate(label_set):
        if (lbl in lbls):
            encoding[idx] = 1
    # normalize due to softmax output
    return encoding/len(lbls)


# get most common image dimension
def get_dimensions():
    dimensions = dict()
    for index, row in df.iterrows():
        img = process_png(row["image_data"])
        dimensions[img.size] = dimensions.get(img.size, 0) + 1
    print (dimensions)
    return dimensions

# df["image_data"].replace(process_png, inplace=True)
df["image_data"] = df["image_data"].apply(process_png)
# df.dropna(subset=["image_data"])
df.dropna(subset=["image_data"], inplace=True)

def encode_row(row):
    return get_encoding(row["Type 1"], row["Type 2"])
df["y"] = df.apply(encode_row, axis=1)

print(df.shape)
print(df.head())