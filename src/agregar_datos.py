import os
import shutil

def agregar_imagen(nueva_imagen_path, destino_folder):
    if not os.path.exists(destino_folder):
        os.makedirs(destino_folder)
    shutil.copy(nueva_imagen_path, destino_folder)

if __name__ == "__main__":
    agregar_imagen('ruta/a/nueva_imagen.jpg', 'dataset/1-50/')
