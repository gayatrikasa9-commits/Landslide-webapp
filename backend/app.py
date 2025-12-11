from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

# ---------------- MODEL LOADING ----------------
# Ensure your model file is actually at this path
MODEL_PATH = "models/unet_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # You might want to handle this more gracefully in a real app

IMG_SIZE = 128

# ---------------- FOLDERS ----------------
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- ROUTES ----------------

# Splash page
@app.route("/")
def splash():
    return render_template("splash.html")

# Home Page (Upload)
@app.route("/home")
def home():
    return render_template("home.html")


# ---------------- PREDICTION ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if not file or file.filename == "":
        return redirect(url_for("home"))

    # Save uploaded image
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Read and preprocess
    img = cv2.imread(img_path)
    if img is None:
        return "Error: Could not read image. Please upload a valid image file."
        
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)

    # Predict mask
    pred_mask = model.predict(img_input)[0]

    # Convert mask properly
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255   # binary mask

    # Save mask
    mask_filename = f"mask_{file.filename}"
    mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)
    cv2.imwrite(mask_path, pred_mask)

    # Create overlay
    # Convert grayscale mask to BGR so it can be added to the original color image
    mask_color = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    # Highlight landslide areas (white in mask) with a red tint in overlay if desired, 
    # or keep it simple. Here we just blend the black/white mask.
    overlay = cv2.addWeighted(img_resized, 0.7, mask_color, 0.3, 0)

    overlay_filename = f"overlay_{file.filename}"
    overlay_path = os.path.join(OUTPUT_FOLDER, overlay_filename)
    cv2.imwrite(overlay_path, overlay)

    # ---------------- LANDSLIDE AREA % ----------------
    landslide_pixels = np.sum(pred_mask == 255)
    total_pixels = pred_mask.size
    percent_affected = (landslide_pixels / total_pixels) * 100

    percent_affected = round(percent_affected, 2)

    # Return results page with correct variables
    return render_template(
        "results.html",
        original=file.filename,
        mask=mask_filename,
        overlay=overlay_filename,
        percent=percent_affected
    )

# ---------------- DOWNLOAD ROUTE (NEW) ----------------
@app.route('/download/<filename>')
def download_file(filename):
    # This allows users to download files from the 'static/outputs' folder
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)