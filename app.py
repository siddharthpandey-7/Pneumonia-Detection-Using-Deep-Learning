from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import base64
import requests
import os
from io import BytesIO

app = Flask(__name__)

# -------------------- HuggingFace Model --------------------
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.h5"
MODEL_PATH = "model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from HuggingFace...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded successfully.")

download_model()

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

IMG_SIZE = (128, 128)

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("result.html", prediction="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("result.html", prediction="No file selected")

    # Save file
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Convert to base64
    pil_image = Image.open(file_path)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    img_data = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    # Preprocess
    img = load_img(file_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    normal, pneumonia = model.predict(arr)[0]

    if pneumonia > normal:
        label = "PNEUMONIA DETECTED"
        confidence = pneumonia * 100
    else:
        label = "NORMAL"
        confidence = normal * 100

    return render_template(
        "result.html",
        prediction=label,
        confidence=round(confidence, 2),
        img_data=img_data
    )

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
