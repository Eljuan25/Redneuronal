# src/cargar_datos.py

import os
import cv2
import numpy as np
import pandas as pd

def cargar_datos(csv_file, img_folder):
    data = pd.read_csv(csv_file)
    images = []
    points = []
    for _, row in data.iterrows():
        img_path = os.path.join(img_folder, row['nombre_imagen'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))  # Redimensionar
        images.append(img)
        points.append((row['x'], row['y']))  # Extraer coordenadas
    return np.array(images), np.array(points)
