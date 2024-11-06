# src/entrenar_modelo.py
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from cargar_datos import cargar_datos

import matplotlib.pyplot as plt

def entrenar_modelo():
    # Cargar los datos
    X, y = cargar_datos('data/labels.csv', 'data/images')
    
    # Verificar que X y y no estén vacíos
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Los datos cargados están vacíos. Verifica la función cargar_datos.")

    print(f'Dimensiones de X: {X.shape}')  # Debe ser algo como (n_samples, 128, 128, 3)
    print(f'Dimensiones de y: {y.shape}')  # Debe ser algo como (n_samples, 2)

    # División de los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo
    model = crear_modelo()

    # Entrenamiento del modelo
    history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    # Graficar la pérdida y precisión del entrenamiento
    plt.figure(figsize=(12, 4))

    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Guardar el modelo entrenado
    model.save('model/modelo_entrenado.h5')
    return model

def crear_modelo():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)  # Salida para coordenadas (x, y)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'accuracy'])
    return model
