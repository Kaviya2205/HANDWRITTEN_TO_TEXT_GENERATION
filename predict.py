import cv2
import numpy as np
import tensorflow as tf
from config import MODEL_PATH, IMAGE_SIZE

def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE) / 255.0
    img = img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    prediction = model.predict(img)
    return np.argmax(prediction)

if __name__ == "__main__":
    image_path = "./test_image.png"  # Provide test image path
    print("Predicted digit:", predict_image(image_path))
