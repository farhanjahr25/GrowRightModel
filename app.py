import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load Keras model
model = tf.keras.models.load_model('model.h5')

# Load TFLite model
with open('model.tflite', 'rb') as f:
    tflite_model = f.read()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the scaler
scaler = joblib.load('normalization_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)

    # Normalize the data
    normalized_data = scaler.transform(input_data)

    # Predict using Keras model
    keras_predictions = model.predict(normalized_data)

    # Set the input tensor for TFLite model
    interpreter.set_tensor(input_details[0]['index'], normalized_data.astype(np.float32))

    # Invoke the interpreter
    interpreter.invoke()

    # Get the output tensor from TFLite model
    tflite_predictions = interpreter.get_tensor(output_details[0]['index'])

    response = {
        'keras_predictions': keras_predictions.tolist(),
        'tflite_predictions': tflite_predictions.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
