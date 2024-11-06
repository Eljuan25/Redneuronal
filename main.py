import sys
import os

# Agregar la ruta absoluta de la carpeta src al sys.path
ruta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(ruta_src)

# Verifica que se haya agregado correctamente
print(f"Ruta agregada al sys.path: {ruta_src}")

# Ahora deberías poder importar los módulos desde la carpeta src
from entrenar_modelo import entrenar_modelo
from visualizar_resultados import visualizar_resultados
from guardar_predicciones import guardar_predicciones

# Entrenar el modelo
modelo = entrenar_modelo()

# Lista de imágenes de prueba
test_images = [
    'data/images/test_image1.jpg',
    'data/images/test_image2.jpg',
    'data/images/test_image3.jpg'
]

# Realizar predicciones en las imágenes de prueba
predicciones = [modelo.predict(cv2.resize(cv2.imread(img), (64, 64)).reshape(1, 64, 64, 3) / 255.0) for img in test_images]

# Guardar las predicciones en un archivo CSV
guardar_predicciones(predicciones, test_images)

# Visualizar los resultados
visualizar_resultados(modelo, test_images)

