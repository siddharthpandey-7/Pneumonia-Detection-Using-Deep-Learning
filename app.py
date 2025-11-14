import os
import requests
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# ---------------- MODEL SETUP ----------------
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.keras"
MODEL_PATH = "best_vgg19_pneumonia.keras"

def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000000:
        print("Downloading model...")
        resp = requests.get(MODEL_URL, stream=True)
        if resp.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            print("Model downloaded.")
        else:
            raise Exception("Could not download model from HF")

model = None
def get_model():
    global model
    if model is None:
        download_model()
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded.")
    return model

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    try:
        img = Image.open(file).convert("RGB").resize((128, 128))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        model_inst = get_model()
        pred = model_inst.predict(arr)

        # softmax outputs = [Normal, Pneumonia]
        prob = float(pred[0][1])
        result = "PNEUMONIA DETECTED" if prob > 0.5 else "NORMAL"
        confidence = round(prob * 100 if prob > 0.5 else (1 - prob) * 100, 2)

        return render_template("result.html",
                               prediction=result,
                               confidence=confidence)

    except Exception as e:
        print("Prediction error:", e)
        return f"Error during prediction: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
