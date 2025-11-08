import os
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Google Drive model file ID
FILE_ID = "1g-M2JrvOxpNCr4hsHJUpqW7EHv3BkYgn"
MODEL_PATH = "best_vgg19_pneumonia.h5"

def download_from_gdrive(file_id, dest_path):
    print("🧠 Model not found locally. Downloading from Google Drive...")
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Sometimes Google requires a confirm token for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print("✅ Model downloaded successfully!")

if not os.path.exists(MODEL_PATH):
    download_from_gdrive(FILE_ID, MODEL_PATH)

# Double-check model validity
if os.path.getsize(MODEL_PATH) < 1000000:
    raise RuntimeError("❌ Model download failed — file too small or invalid. "
                       "Ensure your Google Drive file is shared as 'Anyone with the link'.")

print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

class_names = ['NORMAL', 'PNEUMONIA']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    return render_template('result.html',
                           filename=file.filename,
                           prediction=predicted_class,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
