import os
import subprocess

if __name__ == "__main__":
    if not os.path.exists('models/modelo_entrenado.h5'):
        print("Entrenando el modelo...")
        subprocess.call(["python", "src/entrenamiento.py"])
    
    print("Iniciando la aplicación Flask...")
    subprocess.call(["python", "app.py"])
