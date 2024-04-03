from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def classify_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class probabilities
    predictions = model.predict(img_array)

    # Decode and return the top-3 predicted classes
    decoded_predictions = decode_predictions(predictions, top=7)[0]
    return decoded_predictions

@app.route('/')
def upload_file():
   return render_template('upload.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      f.save(filename)
      result = classify_image(filename)
      os.remove(filename)
      return render_template("result.html", result=result)

if __name__ == '__main__':
   app.run(debug=True)
