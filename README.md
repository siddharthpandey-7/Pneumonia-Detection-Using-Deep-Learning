🩺 Pneumonia Detection using Deep Learning (VGG19 + Flask Web App)

An AI-powered web application that detects Pneumonia from Chest X-ray images using a fine-tuned VGG19 Convolutional Neural Network (CNN).
The model was trained on the Chest X-Ray Images (Pneumonia)
 dataset from Kaggle and integrated into a Flask web application for real-time image-based diagnosis.

🚀 Features

✅ Deep Learning–based Pneumonia detection (VGG19 Transfer Learning)

🧠 Fine-tuned with high test accuracy

🧩 Real-time image upload and prediction using Flask

📸 Beautiful front-end with live preview & loading animation

📊 Confidence score bar and colored prediction output

💾 Model file excluded from GitHub for lightweight repo (Google Drive link provided)

📂 Folder Structure

PNEUMONIA_DETECTION/
│
├── static/
│ ├── style.css # Frontend styling
│ └── uploads/ # Uploaded images (ignored in Git)
│
├── templates/
│ ├── index.html # Upload & preview page
│ └── result.html # Prediction result display
│
├── app.py # Flask backend
├── pneumonia.ipynb # Model training notebook (VGG19)
├── requirements.txt # Required Python libraries
├── .gitignore # Files/folders ignored in Git
└── README.md # Project documentation

🧠 Model Overview

Base Model: VGG19 (pretrained on ImageNet)

Approach: Transfer learning + fine-tuning last convolutional blocks

Input Size: 128×128 RGB images

Optimizer: Adam (lr = 1e-4 → fine-tuned at 1e-5)

Loss Function: Categorical Crossentropy

Epochs: 20 (base) + 10 (fine-tune)

Accuracy: ~95% on test data

Output Classes: NORMAL, PNEUMONIA

🧬 Dataset

Dataset Used: Chest X-Ray Images (Pneumonia) – Kaggle

Training, Validation, and Test splits provided

Data Augmentation applied using ImageDataGenerator

🧠 Download Trained Model

The trained model (best_vgg19_pneumonia.h5) is not uploaded to GitHub due to file size limits.

➡️ Download it from Google Drive:
👉 [Insert your Google Drive model link here]

Once downloaded, place the file in your project root folder next to app.py.

⚙️ Installation & Setup
1️⃣ Clone the Repository

git clone https://github.com/your-username/Pneumonia_Detection.git

cd Pneumonia_Detection

2️⃣ Create a Virtual Environment (optional)

python -m venv venv
venv\Scripts\activate # For Windows

OR

source venv/bin/activate # For Mac/Linux

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run the Application

python app.py

Then open your browser and go to:
🔗 http://127.0.0.1:5000

🌐 Web App Preview
🏠 Upload Page (index.html)

Upload a chest X-ray image

See a live preview

Click Analyze Image

🧾 Result Page (result.html)

Displays the prediction result (PNEUMONIA or NORMAL)

Shows model confidence as a progress bar

“Scan Another Image” button for next prediction

🖥️ Example Output
Input X-ray	Model Prediction
<img src="static/example_normal.jpg" width="200"/>	✅ NORMAL
<img src="static/example_pneumonia.jpg" width="200"/>	⚠️ PNEUMONIA

(Add your real screenshots later here)

🧾 .gitignore Highlights

The following are ignored to keep the repo clean:
venv/
uploads/
*.h5
*.pkl
*.pt
*.joblib
pycache/
*.ipynb_checkpoints
dataset/
data/
chest_xray/

💡 Future Improvements

Add Grad-CAM visualization for model explainability

Deploy web app on Render or Hugging Face Spaces

Add multi-disease classification (Tuberculosis, COVID-19, etc.)

👨‍💻 Author

Siddharth Kumar Pandey
B.Tech – CSE (AI/ML)
📍 India

💼 LinkedIn
 | 🧠 Kaggle
 | 💻 GitHub

🏁 Acknowledgements

Dataset: Kaggle – Chest X-Ray Images (Pneumonia)

Model: VGG19 via TensorFlow/Keras

Framework: Flask

Frontend Design: Custom HTML, CSS, and JavaScript

📜 License

This project is open-source and available under the MIT License.
