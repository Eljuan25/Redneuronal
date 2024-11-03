import os
import subprocess

if __name__ == "__main__":
    model_path = 'models/modelo_entrenado.h5'  # Ruta al modelo

    # Verifica si el modelo ya existe
    if not os.path.exists(model_path):
        print("Entrenando el modelo...")
        # Llama al script de entrenamiento
        result = subprocess.run(["python", "src/entrenamiento.py"])
        
        # Comprueba si el entrenamiento fue exitoso
        if result.returncode != 0:
            print("Error al entrenar el modelo. Asegúrate de que 'entrenamiento.py' funcione correctamente.")
            exit(1)  # Sale si hubo un error en el entrenamiento
    else:
        print("El modelo ya existe. Usando el modelo entrenado.")

    # Inicia la aplicación Flask
    print("Iniciando la aplicación Flask...")
    subprocess.call(["python", "app.py"])
