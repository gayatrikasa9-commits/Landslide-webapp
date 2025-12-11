import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------
# SETTINGS
# ----------------------
IMG_SIZE = 128
MODEL_PATH = "models/unet_model.h5"  # path to your trained model
IMAGE_FOLDER = "dataset/train/images"  # folder with images to test
OUTPUT_FOLDER = "predicted_masks"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------
# LOAD MODEL
# ----------------------
print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ----------------------
# PROCESS AND PREDICT
# ----------------------
image_files = os.listdir(IMAGE_FOLDER)

if len(image_files) == 0:
    print("No images found in folder:", IMAGE_FOLDER)
    exit()

for img_file in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    
    # Load image safely
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: Could not read image {img_file}, skipping")
        continue

    # Resize and normalize
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = img_resized.astype(np.float32)/255.0
    img_input = np.expand_dims(img_input, axis=0)  # add batch dimension

    # Predict mask
    pred_mask = model.predict(img_input)[0]  # remove batch dimension
    pred_mask = (pred_mask * 255).astype(np.uint8)

    # Save mask
    mask_filename = f"mask_{img_file}"
    mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)
    cv2.imwrite(mask_path, pred_mask)
    print(f"Predicted mask saved: {mask_path}")

print("✅ All predictions done!")