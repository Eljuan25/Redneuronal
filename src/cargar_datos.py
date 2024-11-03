import os
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_data(dataset_path, labels_path):
    labels = pd.read_csv(labels_path)
    images = []
    labels_data = []

    for index, row in labels.iterrows():
        img_path = os.path.join(dataset_path, row['filename'])
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))  # Ajustar tamaño
        images.append(image)
        labels_data.append(row['label'])

    images = np.array(images) / 255.0  # Normalizar imágenes
    labels_data = pd.factorize(labels_data)[0]  # Convertir etiquetas a números
    labels_data = to_categorical(labels_data)  # Codificación one-hot

    return images, labels_data
