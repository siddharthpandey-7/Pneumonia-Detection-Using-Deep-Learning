import requests
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model file setup ---
MODEL_PATH = "best_vgg19_pneumonia.h5"
GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?id=1g-M2JrvOxpNCr4hsHJUpqW7EHv3BkYgn"

# Download model if not found locally
if not os.path.exists(MODEL_PATH):
    print("🧠 Model not found locally. Downloading from Google Drive...")
    response = requests.get(GOOGLE_DRIVE_LINK, allow_redirects=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully!")

# --- Load model ---
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- Class names ---
class_names = ['NORMAL', 'PNEUMONIA']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    return render_template('result.html',
                           filename=file.filename,
                           prediction=predicted_class,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
