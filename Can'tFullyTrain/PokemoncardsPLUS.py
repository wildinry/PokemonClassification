# 1. INSTALL NECESSARY LIBRARIES (Run this in your terminal or notebook first)
# !pip install datasets transformers accelerate evaluate torch torchvision Pillow requests

import torch
import numpy as np
import requests
from io import BytesIO
from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import transforms
import evaluate
from PIL import Image
import warnings
import torch.nn.functional as F
from functools import partial

# Suppress the MPS/pin_memory warning that often appears on M-series Macs
warnings.filterwarnings('ignore', category=UserWarning, message=".*pin_memory.*")

# --- CONFIGURATION (STABILIZED FOR M2) ---
DATASET_ID = "bhavnicksm/PokemonCardsPlus"
MODEL_CHECKPOINT = "google/vit-base-patch16-224" 
IMAGE_COLUMN_NAME = "card_image" 
LABEL_COLUMN_NAME = "id"         
OUTPUT_DIR = "pokemon-card-identifier-stable"

# STABILITY SETTINGS:
BATCH_SIZE = 8 
GRADIENT_ACCUMULATION_STEPS = 4
# Set to 0 to prevent multiprocessing errors common on macOS during heavy load
NUM_WORKERS = 0 
# Note: Training will be slower, but it should run without crashing.

NUM_EPOCHS = 5 

# ----------------------------------------------------
# STEP 0: DEFINE ALL FUNCTIONS & CLASSES (Updated)
# ----------------------------------------------------

def load_image_from_url(url):
    """Downloads an image from a URL and returns a PIL Image object."""
    try:
        response = requests.get(url, timeout=10)
        # Note: We handle the PIL warning separately, but stick to RGB for the model
        return Image.open(BytesIO(response.content)).convert("RGB") 
    except Exception:
        # If any download/read fails, return None
        return None

# MODIFIED: The main dataset must be cleaned BEFORE with_transform is applied.
def clean_and_load_data(dataset_split, image_col):
    """Downloads images for the entire dataset and filters out failed samples."""
    print("Pre-downloading and validating images...")
    
    # 1. Map the URLs to image objects (or None)
    # This is done on the main thread, before splitting into workers
    dataset_split = dataset_split.map(
        lambda example: {'image_object': load_image_from_url(example[image_col])},
        num_proc=1, # Keep this sequential to avoid multi-processing issues on initial load
    )
    
    # 2. Filter out all samples where the image failed to load (image_object is None)
    initial_count = len(dataset_split)
    dataset_split = dataset_split.filter(lambda example: example['image_object'] is not None)
    final_count = len(dataset_split)

    print(f"Cleaned dataset: {initial_count} samples -> {final_count} valid samples.")
    return dataset_split

# MODIFIED: Preprocessing now receives the image_object directly
def preprocess_dataset(example_batch, transforms_fn, label2id, label_col):
    # This function expects 'image_object' to already be a valid PIL image
    images = example_batch['image_object']
    
    # Apply transforms and map string labels to numerical IDs
    example_batch["pixel_values"] = [transforms_fn(img) for img in images]
    example_batch["label"] = [label2id[label] for label in example_batch[label_col]]
    
    return example_batch

def compute_metrics(eval_pred):
    """Computes accuracy during model evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1) 
    return accuracy_metric.compute(predictions=predictions, references=labels)

def collate_fn(examples):
    """Ensures the input pixel values are stacked correctly into a batch tensor."""
    # We now access 'pixel_values' and 'label' directly as they were created by with_transform
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# ==============================================================================
# WRAP ALL EXECUTION LOGIC IN THE MAIN GUARD
# ==============================================================================
if __name__ == '__main__':
    
    # ----------------------------------------------------
    # STEP 1: LOAD DATASET AND CREATE LABEL MAPPINGS
    # ----------------------------------------------------
    print("Step 1: Loading Dataset and Creating Label Mappings...")
    dataset = load_dataset(DATASET_ID)
    
    # Clean the dataset before splitting to ensure the train/val split is robust
    cleaned_dataset = clean_and_load_data(dataset["train"], IMAGE_COLUMN_NAME)
    
    # Split the cleaned dataset
    dataset_split = cleaned_dataset.train_test_split(test_size=0.1, seed=42)

    labels = dataset_split["train"].unique(LABEL_COLUMN_NAME)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    print(f"Total unique Pokémon cards (classes): {num_labels}")


    # ----------------------------------------------------
    # STEP 2: DATA PREPROCESSING AND TRANSFORMS
    # ----------------------------------------------------
    print("Step 2: Configuring Preprocessor and Transforms...")
    image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

    normalize = transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    image_size = image_processor.size["height"]

    _train_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    _val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    # Use functools.partial to safely inject variables
    train_transform_fn = partial(
        preprocess_dataset,
        transforms_fn=_train_transforms,
        label2id=label2id,
        label_col=LABEL_COLUMN_NAME
    )
    
    val_transform_fn = partial(
        preprocess_dataset,
        transforms_fn=_val_transforms,
        label2id=label2id,
        label_col=LABEL_COLUMN_NAME
    )

    # Apply the new simplified transforms
    train_dataset = dataset_split["train"].with_transform(train_transform_fn)
    val_dataset = dataset_split["test"].with_transform(val_transform_fn)


    # ----------------------------------------------------
    # STEP 3: MODEL AND METRICS SETUP
    # ----------------------------------------------------
    print("Step 3: Setting up Model and Metrics...")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True 
    )

    # CRITICAL STABILITY FIX: Disable torch.compile to prevent SegFaults
    # if torch.backends.mps.is_available():
    #     model = torch.compile(model, backend="aot_eager")
    #     print("✅ Model compiled using torch.compile for M2 acceleration.")

    accuracy_metric = evaluate.load("accuracy")


    # ----------------------------------------------------
    # STEP 4: TRAINER CONFIGURATION AND TRAINING
    # ----------------------------------------------------
    print("Step 4: Configuring Trainer and Starting Training...")

    if torch.backends.mps.is_available():
        print("Using Apple Silicon (MPS) for acceleration.")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",  
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=2e-5, 
        save_total_limit=1,
        remove_unused_columns=False,
        # CRITICAL STABILITY FIX: Disable fp16 (half-precision)
        # Switch to full precision (fp32) for better stability on MPS
        fp16=False, 
        report_to="none", 
        
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # CRITICAL STABILITY FIX: Use NUM_WORKERS=0 
        dataloader_num_workers=NUM_WORKERS,
        load_best_model_at_end=True, 
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # ----------------------------------------------------
    # STEP 5: PREDICTION AND ACCURACY CHECK
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("STEP 5: PREDICTION AND ACCURACY CHECK")
    print("="*50)

    final_results = trainer.evaluate()
    print(f"Final Validation Accuracy: {final_results.get('eval_accuracy', 'N/A'):.4f}")

    # --- Sample Prediction ---
    sample_index = 0
    # Access the sample from the raw split to get the original URL for display
    raw_sample = dataset_split["test"][sample_index]
    sample = val_dataset[sample_index]
    true_label_id = id2label[sample['label']]
    true_label_url = raw_sample[IMAGE_COLUMN_NAME]

    print("\n--- Single Card Prediction Sample ---")
    print(f"True Card ID: {true_label_id}")
    print(f"Image Source: {true_label_url}")


    # Prepare the sample for the model
    input_tensor = sample['pixel_values'].unsqueeze(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model.to(device) 

    # Get predictions (logits)
    with torch.no_grad():
        output = model(input_tensor)
        logits = output.logits

    # Convert logits to probabilities and get top 5 predictions
    probabilities = F.softmax(logits, dim=1)
    top_p, top_class = probabilities.topk(5, dim=1)

    print("\nTop 5 Predictions:")
    for i in range(5):
        predicted_id = id2label[top_class.squeeze().tolist()[i]]
        confidence = top_p.squeeze().tolist()[i] * 100
        
        print(f"  {i+1}. ID: {predicted_id} (Confidence: {confidence:.2f}%)")
