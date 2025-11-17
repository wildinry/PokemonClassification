
#Pokémon Type Classifier (CNN Model)

This project trains a Convolutional Neural Network (CNN) to classify Pokémon by Type1 using image data.
The dataset consists of pokemon.csv and an accompanying folder of Pokémon images.

###Features

Loads and preprocesses Pokémon images

Encodes type labels for classification

Trains a CNN model using TensorFlow/Keras

Saves the model and label encoder

Includes a prediction script (predict.py) to classify new images

##Getting Started
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

##2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

##3. Install dependencies
pip install -r requirements.txt

##4. Extract the images

A compressed image folder is included in the repository. Extract it before training:

unzip images_compressed.zip -d images/


Your project structure should look like:

project/
│── main.py
│── predict.py
│── pokemon.csv
│── images/
│     bulbasaur.png
│     charmander.png
│     squirtle.png
│     ...
│── requirements.txt

##Training the Model

Run the training script:

python main.py


This script will:

Load the data from pokemon.csv

Load and preprocess all Pokémon images

Train the convolutional neural network

Save the following files:

pokemon_model.h5

label_classes.npy

Note: The model file is large and is not included in the repository.

Making Predictions

Use the prediction script after training is complete:

python predict.py images/pikachu.png


##Sample output:

images/pikachu.png → Predicted Type1: Electric


You may substitute any valid image path.

File Overview
File	Description
main.py	Trains the CNN and saves the model and label encoder
predict.py	Loads the model and predicts the Pokémon type
pokemon.csv	Pokémon metadata including Type1
images/	Pokémon image files
requirements.txt	Python dependencies
Notes

Training time depends on your hardware. GPU acceleration is recommended.

Image files must match the Pokémon names in pokemon.csv and be lowercase, e.g.:

bulbasaur.png

charmander.png

squirtle.png

License

This project is released under the MIT License.
