import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf  # Ajoutez cette ligne pour importer TensorFlow

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display, clear_output
import time

print(tf.__version__)  # Affiche la version de TensorFlow
# Définir les chemins vers les données
train_path = '/home/cedric/Bureau/Travail/Arts Plastique/archive(1)/images/images/train'
test_path = '/home/cedric/Bureau/Travail/Arts Plastique/archive(1)/images/images/validation'
categories = ['angry', "disgust", "fear", 'happy', "neutral", "sad", "surprise"]

# Fonction pour charger et pré-traiter les données
def load_data(data_path, categories, img_size=(128, 128)):
    data, labels = [], []
    for idx, category in enumerate(categories):
        folder_path = os.path.join(data_path, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                data.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Erreur avec l'image {img_name}: {e}")
    return np.array(data), np.array(labels)

# Charger les données d'entraînement et de test
x_train, y_train = load_data(train_path, categories)
x_test, y_test = load_data(test_path, categories)

# Normalisation et encodage des labels
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes=len(categories))
y_test = to_categorical(y_test, num_classes=len(categories))

# Définition du modèle avec Dropout
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compiler et entraîner le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

datagen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=10,
    callbacks=[early_stopping]
)

# Évaluation du modèle
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Matrice de confusion
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)

# Affichage de la matrice de confusion
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.show()

# Générer et afficher le nuage de mots
predictions = np.argmax(model.predict(x_test), axis=1)
word_counts = {categories[i]: (predictions == i).sum() for i in range(len(categories))}
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Nuage de Mots des Prédictions")
plt.show()

# Affichage des prédictions pour 100 images d'essai
for i in range(100):
    img = x_test[i]
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = categories[np.argmax(pred)]
    
    plt.imshow(img)
    plt.title(f"Classe Prédite : {pred_class}")
    plt.axis('off')
    display(plt.gcf())
    time.sleep(1)
    clear_output(wait=True)
