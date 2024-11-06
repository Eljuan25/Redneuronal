import pandas as pd
import os
from tensorflow.keras.preprocessing import image
import numpy as np

def cargar_datos(labels_path, images_path):
    try:
        # Cargar las etiquetas desde el archivo CSV
        labels = pd.read_csv(labels_path)
    except FileNotFoundError:
        print(f"Error: El archivo de etiquetas no se encontró en {labels_path}")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: El archivo CSV está vacío.")
        return None, None
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return None, None

    labels['image'] = labels['image'].astype(str)

    image_list = []
    
    for img_name in labels['image']:
        img_path = os.path.join(images_path, img_name)
        
        if os.path.exists(img_path):
            try:
                img = image.load_img(img_path, target_size=(128, 128))
                img_array = image.img_to_array(img)
                image_list.append(img_array)
            except Exception as e:
                print(f"Error al cargar la imagen {img_name}: {e}")
        else:
            print(f"Advertencia: La imagen {img_name} no se encontró en {images_path}")
    
    if len(image_list) == 0:
        print("Error: No se cargaron imágenes.")
        return None, None
    X = np.array(image_list)
    
    try:
        y = labels[['x', 'y']].values
    except KeyError:
        print("Error: Las columnas 'x' y 'y' no se encontraron en el archivo CSV.")
        return None, None
    
    if X.shape[0] == 0 or y.shape[0] == 0:
        print("Error: No se cargaron imágenes o etiquetas.")
        return None, None
    
    return X, y
