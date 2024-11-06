# src/entrenar_modelo.py

import tensorflow as tf
from tensorflow.keras import layers, models
from cargar_datos import cargar_datos

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
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def entrenar_modelo():
    X, y = cargar_datos('data/labels.csv', 'data/images')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = crear_modelo()
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
    model.save('model/modelo_entrenado.h5')
    return model
