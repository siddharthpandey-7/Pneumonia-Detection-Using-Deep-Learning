from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests

# --- Initialize Flask app ---
app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model path and Google Drive download link ---
MODEL_PATH = 'best_vgg19_pneumonia.h5'

# 🔗 Replace the below link with your direct Google Drive download link
# Example: https://drive.google.com/uc?id=YOUR_FILE_ID
GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?id=1g-M2JrvOxpNCr4hsHJUpqW7EHv3BkYgn"

# --- Download model automatically if not found ---
if not os.path.exists(MODEL_PATH):
    print("🧠 Model not found locally. Downloading from Google Drive...")
    response = requests.get(GOOGLE_DRIVE_LINK, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully.")
    else:
        print("❌ Failed to download model. Please check the Google Drive link.")
else:
    print("✅ Model found locally, skipping download.")

# --- Load trained model ---
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# --- Class names ---
class_names = ['NORMAL', 'PNEUMONIA']

# --- Homepage route ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    return render_template('result.html',
                           filename=file.filename,
                           prediction=predicted_class,
                           confidence=confidence)

# --- Display uploaded image ---
@app.route('/display/<filename>')
def display_image(filename):
    return f'<img src="/static/uploads/{filename}" width="300">'

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
