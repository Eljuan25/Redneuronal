from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Ruta correcta al modelo, ajustada para que esté fuera de la carpeta dataset
model_path = 'models/modelo_entrenado.h5'

# Verifica si el modelo existe antes de cargarlo
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: El archivo del modelo no se encuentra en la ruta especificada: {model_path}")

# Carga el modelo
try:
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        # Decodificar la imagen desde el archivo
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format.'}), 400
        
        img = cv2.resize(img, (224, 224)) / 255.0  # Preprocesar imagen
        img = np.expand_dims(img, axis=0)  # Agregar dimensión batch
        
        # Realizar la predicción
        preds = model.predict(img)
        predicted_class = np.argmax(preds[0])  # Obtener la clase predicha
        return jsonify({'class': str(predicted_class)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


