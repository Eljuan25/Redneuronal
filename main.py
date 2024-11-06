import sys
import os
import zipfile
import cv2

# Asegurarse de agregar la ruta absoluta de la carpeta src al sys.path
ruta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(ruta_src)

# Verifica que se haya agregado correctamente
print(f"Ruta agregada al sys.path: {ruta_src}")

# Ahora, importa las funciones desde src
from src.entrenar_modelo import cargar_datos
from src.entrenar_modelo import entrenar_modelo
from src.visualizar_resultados import visualizar_resultados
from src.guardar_predicciones import guardar_predicciones

# Ruta del archivo CSV con las etiquetas
labels_path = 'data/labels.csv'

# Ruta de las imágenes
images_path = 'data/images'

# Llamar a la función para cargar datos
X, y = cargar_datos(labels_path, images_path)

if X is not None and y is not None:
    print(f"Se cargaron correctamente {X.shape[0]} imágenes.")
else:
    print("Error al cargar las imágenes y/o etiquetas.")

# Descomprimir el archivo ZIP si es necesario
zip_path = '1-50.zip'
extract_to = 'data/images'
if os.path.exists(zip_path) and not os.listdir(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Imágenes extraídas a: {extract_to}")
else:
    print("Archivo ZIP no encontrado o las imágenes ya están descomprimidas.")

# Entrenar el modelo
modelo = entrenar_modelo()

# Lista de imágenes de prueba
test_images = [
    'data/images/test_image1.jpg',
    'data/images/test_image2.jpg',
    'data/images/test_image3.jpg'
]

# Realizar predicciones en las imágenes de prueba
predicciones = []
for img_path in test_images:
    # Cargar y preprocesar cada imagen
    img = cv2.imread(img_path)
    if img is not None:
        # Redimensionar y normalizar la imagen
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized.reshape(1, 64, 64, 3) / 255.0
        # Realizar la predicción
        prediccion = modelo.predict(img_normalized)
        predicciones.append(prediccion)
    else:
        print(f"Advertencia: La imagen {img_path} no se encontró o no se pudo cargar.")

# Guardar las predicciones en un archivo CSV
guardar_predicciones(predicciones, test_images)

# Visualizar los resultados
for img_path, prediccion in zip(test_images, predicciones):
    visualizar_resultados(modelo, img_path, prediccion)
