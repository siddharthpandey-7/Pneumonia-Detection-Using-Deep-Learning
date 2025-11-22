from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import base64
import os
from io import BytesIO

app = Flask(__name__)

MODEL_PATH = "best_vgg19_pneumonia.h5"
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

IMG_SIZE = (128, 128)

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

    # Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Convert image to base64 for display
    pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_data = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

    # Preprocess
    img = load_img(file_path, target_size=IMG_SIZE)
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_arr)[0]
    normal_prob, pneumonia_prob = prediction

    if pneumonia_prob > normal_prob:
        label = "PNEUMONIA DETECTED"
        confidence = pneumonia_prob * 100
    else:
        label = "NORMAL"
        confidence = normal_prob * 100

    confidence = round(confidence, 2)

    return render_template(
        "result.html",
        prediction=label,
        confidence=confidence,
        img_data=img_data   # <-- IMPORTANT
    )

if __name__ == "__main__":
    app.run(debug=True)
