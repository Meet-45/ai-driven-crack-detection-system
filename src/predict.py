# predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import IMG_SIZE, MODEL_PATH

def predict_image(image_path):
    """
    Predicts whether an image contains a crack or not
    """

    model = load_model(MODEL_PATH)

    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    return "Crack" if prediction >= 0.5 else "No Crack"


if __name__ == "__main__":
    result = predict_image("sample.jpg")
    print("Prediction:", result)
