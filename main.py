import sys
import os

# Agregar la ruta absoluta de la carpeta src al sys.path
ruta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(ruta_src)

# Verifica que se haya agregado correctamente
print(f"Ruta agregada al sys.path: {ruta_src}")

# Ahora deberías poder importar el módulo
from modelo_entrenado import entrenar_modelo

