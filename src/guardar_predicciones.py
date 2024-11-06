import csv

def guardar_predicciones(predicciones, nombres_imagenes, archivo_salida='resultados/predicciones.csv'):
    with open(archivo_salida, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Imagen', 'Predicción'])  # Cabeceras de columna

        # Escribir cada imagen y su predicción
        for nombre_imagen, prediccion in zip(nombres_imagenes, predicciones):
            writer.writerow([nombre_imagen, prediccion])

    print(f"Predicciones guardadas en {archivo_salida}")
