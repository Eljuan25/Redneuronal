# src/visualizar_resultados.py

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from cargar_datos import cargar_datos

def visualizar_predicciones(model, X, y):
    predicciones = model.predict(X)
    for i in range(5):  # Muestra 5 imágenes de ejemplo
        plt.imshow(X[i].astype("uint8"))
        plt.scatter(y[i][0], y[i][1], color='green', label='Punto Correcto')
        plt.scatter(predicciones[i][0], predicciones[i][1], color='red', label='Predicción')
        plt.legend()
        plt.show()

def main():
    X, y = cargar_datos('data/labels.csv', 'data/images')
    model = load_model('model/modelo_entrenado.h5')
    visualizar_predicciones(model, X, y)
