import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import cv2

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="D:/Vibe/detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define disease classes
DISEASE_CLASSES = [
    "Healthy",
    "Respiratory Disease",
    "Skin Disease",
    "Gastrointestinal Disease"
]

def preprocess_image(image):
    # Resize image to match model input shape
    input_shape = input_details[0]['shape']
    img = cv2.resize(image, (input_shape[1], input_shape[2]))
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    confidence = float(output_data[0][prediction])
    
    return {
        'disease': DISEASE_CLASSES[prediction],
        'confidence': confidence
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and convert image
        img = Image.open(file.stream).convert('RGB')
        img_array = np.array(img)
        
        # Get prediction
        result = predict_disease(img_array)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 