import numpy as np
from cargar_datos import load_data
from modelo_cnn import create_model

if __name__ == "__main__":
    # Cargar datos
    images, labels = load_data('dataset/1-50/', 'dataset/labels.csv')

    # Crear el modelo
    model = create_model(input_shape=(224, 224, 3), num_classes=labels.shape[1])

    # Entrenar el modelo
    model.fit(images, labels, epochs=10, validation_split=0.2)  # Ajusta epochs según sea necesario

    # Guardar el modelo
    model.save('models/modelo_entrenado.h5')
