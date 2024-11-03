import numpy as np
import cv2
from tensorflow.keras.models import load_model

def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0  # Preprocesar imagen
    img = np.expand_dims(img, axis=0)  # Agregar dimensión batch
    preds = model.predict(img)
    predicted_class = np.argmax(preds[0])  # Obtener la clase predicha
    return predicted_class

if __name__ == "__main__":
    model = load_model('models/modelo_entrenado.h5')
    predicted_class = predict_image(model, 'ruta/a/la/imagen.jpg')
    print(f'Clase predicha: {predicted_class}')
