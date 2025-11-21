##Pokémon Type Classifier (CNN Model)

|**Task** | **Model** | **Libraries** |
| :---: | :---: | :---: |
| Image Classification | Convolutional Neural Network ($\text{CNN}$) | TensorFlow, Keras, NumPy, Pandas |

This project trains a **Convolutional Neural Network (CNN)** to classify Pokémon based on their primary type ($\text{Type1}$) using image data.

---

### Features

**Image Preprocessing:** Efficiently loads and processes Pokémon sprite images for $\text{CNN}$ input.
**Label Encoding:** Handles categorical $\text{Type1}$ labels for multi-class classification training.
**Deep Learning:** Trains a robust $\text{CNN}$ model built with **TensorFlow/Keras**.
**Persistence:** Automatically saves the trained model (`pokemon_model.h5`) and the label mapping (`label_classes.npy`).
**Prediction Utility:** Includes a dedicated script (`predict.py`) for easy classification of new images.

---

### Getting Started

Follow these steps to set up your environment and prepare the training data.

#### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/wildinry/ACO494_Final_Pokemon_Image.git)
cd YOUR_REPO
```
#### 2. Setup Environment
Create a virtual environment and activate it to manage dependencies:
```bash
python3 -m venv venv
# For Mac/Linux:
source venv/bin/activate
# For Windows:
venv\Scripts\activate
```
#### 3. Install Requirements
Install all necessary Python packages:
```bash
pip install -r requirements.txt
```
#### 4. Clone the Repository
The images are compressed. Unzip them into the expected images/ directory:
```bash
unzip imageCompress.zip 
#for Windows:
tar -xf imageCompress.zip
```
[!NOTE] Image Naming Convention: Image files must match the Pokémon names found in pokemon.csv and be all lowercase (e.g., bulbasaur.png, pikachu.png).
#### Project Structure
A successful setup will result in the following key file structure:
```bash
project/
├── MobileVnetImplementation.py  
├── pokemon.csv                  # Pokémon metadata (Type1, etc.)
├── predict.py                   # Script for running predictions
├── ReadME.md
├── requirements.txt             # Python dependencies
├── simpleCNN.py                 # Main script for training the CNN model
├── vit.py
└── images/                      # Directory containing all unzipped Pokémon image files
    ├── bulbasaur.png
    └── ... (all images)
```

#### Training the Model
Execute the primary script to begin training:
```bash
python main.py
```
Upon completion, the script will generate and save these files:

pokemon_model.h5

label_classes.npy

[!WARNING] Performance Tip Training time can be significant. GPU acceleration is strongly recommended for faster results. The pokemon_model.h5 file is large and is generally excluded from the Git repository.

#### Making Predictions
Use predict.py and provide a path to the image you wish to classify:

```bash
python predict.py images/pikachu.png
```

#### Example Output

```bash
images/pikachu.png → Predicted Type1: Electric
```
Substitute images/pikachu.png with any valid image path from your dataset.
