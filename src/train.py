# train.py

from data_loader import get_data_generators
from model_builder import build_cnn_model
from config import EPOCHS, MODEL_PATH
import matplotlib.pyplot as plt

def train():
    train_gen, val_gen = get_data_generators()

    model = build_cnn_model()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1
    )

    # Save trained model
    model.save(MODEL_PATH)
    print("Model saved successfully.")

    # Plot accuracy & loss
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

if __name__ == "__main__":
    train()
