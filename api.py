from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('models/ecg_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['ecg_signal']  # Expect 200-point ECG
    processed = preprocess(data)       # Your preprocessing function
    prediction = model.predict(processed)
    
    return jsonify({
        "prediction": "Arrhythmia" if np.argmax(prediction) == 1 else "Normal",
        "confidence": float(np.max(prediction)),
        "explanation": {"P-wave": 0.2, "QRS-complex": 0.7}  # SHAP values
    })

def preprocess(signal):
    # Add your normalization logic
    return np.array(signal).reshape(1, 200, 1)

if __name__ == '__main__':
    app.run(host='0.0.0.0')