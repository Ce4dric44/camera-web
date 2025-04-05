from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Charger le modèle .h5
model = load_model("/home/cedric/Bureau/Travail/Arts Plastique/mon_modele_final.h5")

# Liste des émotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Fonction pour prédire l'émotion
def predict_emotion(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)  # Ajouter la dimension du canal de couleur
    image = np.expand_dims(image, axis=0)   # Ajouter la dimension du batch
    prediction = model.predict(image)
    emotion_index = np.argmax(prediction, axis=-1)[0]
    return emotions[emotion_index]

# Route API pour recevoir l'image et prédire l'émotion
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer l'image envoyée dans la requête
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Prédire l'émotion
        emotion = predict_emotion(img)
        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

# Lancer le serveur Flask automatiquement
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Désactive le reloader pour éviter qu'il ne redémarre le serveur en boucle
