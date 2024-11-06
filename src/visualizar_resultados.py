import matplotlib.pyplot as plt
import cv2

def visualizar_resultados(model, test_images):
    for img_path in test_images:
        # Cargar la imagen
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para matplotlib

        # Preprocesar la imagen para hacer la predicción
        img_resized = cv2.resize(img, (64, 64))  # Ajustar el tamaño según el modelo
        img_input = img_resized / 255.0  # Normalizar
        img_input = img_input.reshape(1, 64, 64, 3)  # Asegurar la dimensión correcta

        # Realizar la predicción
        prediccion = model.predict(img_input)

        # Mostrar la imagen y el resultado de la predicción
        plt.imshow(img)
        plt.title(f"Predicción: {prediccion}")
        plt.show()
