import tensorflow as tf
from preprocess import load_dataset
from model import create_model
from config import MODEL_PATH, EPOCHS, BATCH_SIZE

def train_model():
    X_train, X_test, y_train, y_test = load_dataset()
    model = create_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save(MODEL_PATH)
    print("Model saved at", MODEL_PATH)

if __name__ == "__main__":
    train_model()
