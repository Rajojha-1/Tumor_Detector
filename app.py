from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Initialize app
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model.h5')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction=None, accuracy=None)
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction=None, accuracy=None)
    if file and allowed_file(file.filename):
        img = Image.open(file).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        return render_template('index.html', prediction=predicted_class, accuracy=confidence)
    else:
        return render_template('index.html', prediction=None, accuracy=None)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

