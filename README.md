⚡ Pokémon Type Classifier (CNN Model)This project trains a Convolutional Neural Network (CNN) to classify Pokémon by their primary type ($\text{Type1}$) using image data.The project uses a structured dataset consisting of pokemon.csv and an accompanying folder of Pokémon images for training and classification.✨ FeaturesLoads and preprocesses Pokémon images for $\text{CNN}$ input.Encodes type labels for multi-class classification.Trains a $\text{CNN}$ model using TensorFlow/Keras.Saves the trained model (pokemon_model.h5) and the label encoder (label_classes.npy).Includes a dedicated prediction script (predict.py) to classify new Pokémon images.🚀 Getting StartedFollow these steps to set up the environment and prepare the data.1. Clone the RepositoryBashgit clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
2. Create and Activate a Virtual EnvironmentIt's highly recommended to use a virtual environment to manage dependencies.Bashpython3 -m venv venv
# For Mac/Linux:
source venv/bin/activate
# For Windows:
venv\Scripts\activate
3. Install DependenciesInstall all required Python packages:Bashpip install -r requirements.txt
4. Extract the ImagesA compressed image folder is included in the repository. Extract its contents before starting the training process.Bashunzip images_compressed.zip -d images/
📂 Project StructureYour project directory should now look like this:project/
├── main.py
├── predict.py
├── pokemon.csv
├── images/
│   ├── bulbasaur.png
│   ├── charmander.png
│   ├── squirtle.png
│   └── ...
└── requirements.txt
🤖 Training the ModelRun the main script to train the model:Bashpython main.py
This script will perform the following actions:Load the data from pokemon.csv.Load and preprocess all Pokémon images.Train the convolutional neural network.Save the following files:pokemon_model.h5label_classes.npyNote: Training time depends heavily on your hardware. GPU acceleration is highly recommended. The model file (pokemon_model.h5) is large and is not included in the repository.🔎 Making PredictionsAfter training is complete, you can use the prediction script to classify a new image:Bashpython predict.py images/pikachu.png
Sample Output:images/pikachu.png → Predicted Type1: Electric
You can substitute images/pikachu.png with any valid image path you want to classify.
