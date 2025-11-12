import os
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Disable oneDNN optimization (Render CPU optimization causes bias)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# ✅ Hugging Face Model URL
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.h5"
MODEL_PATH = "best_vgg19_pneumonia.h5"

# ------------------ DOWNLOAD MODEL ------------------
def download_model():
    """Download model from Hugging Face if not already present."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        print("🧠 Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            raise Exception("❌ Failed to download model from Hugging Face.")

# ------------------ LOAD MODEL ------------------
model = None
def get_model():
    """Lazy load model only once."""
    global model
    if model is None:
        download_model()
        print("🧩 Loading model into memory...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
    return model

# ------------------ ROUTES ------------------
@app.route("/")
def home():
    """Render homepage."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction."""
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    try:
        # Save uploaded file temporarily
        upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)

        # Preprocess image (match training size)
        img = Image.open(filepath).convert("RGB").resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        model_instance = get_model()
        prediction = model_instance.predict(img_array)
        prob = float(prediction[0][0])
        confidence = round(prob * 100, 2) if prob > 0.5 else round((1 - prob) * 100, 2)
        result = "PNEUMONIA DETECTED" if prob > 0.5 else "NORMAL"

        print(f"✅ Prediction: {result}, Confidence: {confidence}%")

        return render_template(
            "result.html",
            prediction=result,
            confidence=confidence,
            filename=file.filename
        )

    except Exception as e:
        print("❌ Prediction error:", e)
        return f"Error during prediction: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
