import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import requests
from io import BytesIO

# Set the training directory path
train_dir = 'data/dataset/splits/train'

# Dynamically create disease_names list based on the folder names in the training directory
disease_names = sorted(os.listdir(train_dir))

def load_image(img_path):
    if img_path.startswith('http'):
        response = requests.get(img_path)
        img = image.load_img(BytesIO(response.content), target_size=(224, 224))
    else:
        img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict(image_path):
    try:
        model = load_model('model/fungal_skin_model_resnet50v2.keras')
        img_array = load_image(image_path)

        # Suppress TensorFlow progress bar
        predictions = model.predict(img_array)
        
        decoded_predictions = np.argmax(predictions, axis=1)

        if decoded_predictions[0] < len(disease_names):
            disease_name = disease_names[decoded_predictions[0]]
        else:
            disease_name = "Unknown"

        accuracy = predictions[0][decoded_predictions[0]] * 100
        result = {
            "disease": disease_name,
            "accuracy": round(accuracy, 2)
        }
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    image_path = sys.argv[1]
    result = predict(image_path)
    print(json.dumps(result, ensure_ascii=False))
