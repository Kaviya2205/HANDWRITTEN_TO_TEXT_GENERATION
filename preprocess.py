import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from config import DATASET_PATH, IMAGE_SIZE, NUM_CLASSES

def load_dataset():
    images, labels = [], []
    for label in range(NUM_CLASSES):
        folder_path = os.path.join(DATASET_PATH, str(label))
        for file in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMAGE_SIZE) / 255.0
            images.append(img)
            labels.append(label)
    images = np.array(images).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    labels = to_categorical(labels, NUM_CLASSES)
    return train_test_split(images, labels, test_size=0.2, random_state=42)
