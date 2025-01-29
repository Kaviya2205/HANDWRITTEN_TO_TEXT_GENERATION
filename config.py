##### config.py #####
import os

# Dataset path
DATASET_PATH = "./dataset/"
MODEL_PATH = "./saved_model/handwriting_model.h5"
IMAGE_SIZE = (28, 28)  # Resizing to 28x28 pixels
NUM_CLASSES = 10  # Update based on dataset
BATCH_SIZE = 32
EPOCHS = 10
