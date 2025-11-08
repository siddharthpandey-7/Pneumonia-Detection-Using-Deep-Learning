import os
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# --- Initialize Flask app ---
app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Google Drive model details ---
MODEL_PATH = "best_vgg19_pneumonia.h5"
GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?export=download&id=1g-M2JrvOxpNCr4hsHJUpqW7EHv3BkYgn"

# --- Function to download model from Google Drive ---
def download_model():
    print("🧠 Model not found locally. Downloading from Google Drive...")
    response = requests.get(GOOGLE_DRIVE_LINK, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("✅ Model downloaded successfully!")

# --- Download model if not found ---
if not os.path.exists(MODEL_PATH):
    download_model()

# --- Verify model file integrity ---
if os.path.getsize(MODEL_PATH) < 1000000:  # less than 1MB means it's invalid (HTML file)
    raise RuntimeError(
        "❌ Model download failed. File is too small or invalid.\n"
        "👉 Please ensure the file is shared as 'Anyone with the link' in Google Drive."
    )

# --- Load the trained model ---
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- Class labels (same order as training) ---
class_names = ['NORMAL', 'PNEUMONIA']

# --- Routes ---
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

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image for model
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    return render_template(
        'result.html',
        filename=file.filename,
        prediction=predicted_class,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
