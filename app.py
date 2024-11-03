from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('models/modelo_entrenado.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0  # Preprocesar imagen
    img = np.expand_dims(img, axis=0)  # Agregar dimensión batch
    preds = model.predict(img)
    predicted_class = np.argmax(preds[0])  # Obtener la clase predicha
    return jsonify({'class': str(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
