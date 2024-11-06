import cv2
import matplotlib.pyplot as plt

def visualizar_resultados(modelo, img_path, puntos):
    # Cargar la imagen
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Dibujar los puntos sobre la imagen
    for punto in puntos:
        x, y = punto["x"], punto["y"]
        nombre_punto = punto["nombre_punto"]
        cv2.circle(img_rgb, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(img_rgb, nombre_punto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Mostrar la imagen con puntos
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
