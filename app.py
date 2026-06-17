from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import center_of_mass, zoom
import base64
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_digit_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

def preprocess_canvas(img_bytes):
    """
    Mimics MNIST preprocessing:
    1. Grayscale + invert (white digit on black)
    2. Crop tight to digit bounding box
    3. Thin thick strokes via erosion-like resize
    4. Resize to 20x20 preserving aspect ratio
    5. Pad to 28x28 with digit centered at center of mass (MNIST style)
    6. Normalize
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32)

    # Canvas: white strokes on black bg — already correct for MNIST
    # But browser canvas draws white on black here, so no invert needed
    # Threshold to clean noise
    arr = np.where(arr > 30, arr, 0)

    # --- 1. Crop tight to bounding box ---
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if not rows.any():
        # Empty canvas
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = arr[rmin:rmax+1, cmin:cmax+1]

    # --- 2. Resize to 20x20 preserving aspect ratio ---
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    cropped_img = Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(cropped_img, dtype=np.float32)

    # --- 3. Place on 28x28 canvas, centered at center of mass (MNIST style) ---
    canvas = np.zeros((28, 28), dtype=np.float32)
    # Center of mass of the resized digit
    cy, cx = center_of_mass(resized)
    if np.isnan(cy): cy = new_h / 2
    if np.isnan(cx): cx = new_w / 2

    # Target: center of mass should land at (14, 14)
    top  = int(round(14 - cy))
    left = int(round(14 - cx))

    # Clamp so it fits
    top  = max(0, min(top,  28 - new_h))
    left = max(0, min(left, 28 - new_w))

    canvas[top:top+new_h, left:left+new_w] = resized

    # --- 4. Normalize to [0, 1] ---
    canvas = canvas / 255.0

    return canvas.reshape(1, 28, 28, 1)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        img_array = preprocess_canvas(img_bytes)

        predictions = model.predict(img_array, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        all_probs = [round(float(p) * 100, 1) for p in predictions]

        return jsonify({
            "digit": predicted_digit,
            "confidence": round(confidence * 100, 1),
            "probabilities": all_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Running on http://localhost:5000")
    app.run(debug=True, port=5000)