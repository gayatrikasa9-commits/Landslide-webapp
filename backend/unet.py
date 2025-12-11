import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
IMG_SIZE = 128
images = []
masks = []

img_dir = "dataset/train/images"
mask_dir = "dataset/train/masks"

# Get sorted list of files
img_files = sorted(os.listdir(img_dir))
mask_files = sorted(os.listdir(mask_dir))

# Make sure the number of images and masks match
num_files = min(len(img_files), len(mask_files))

for i in range(num_files):
    img_path = os.path.join(img_dir, img_files[i])
    mask_path = os.path.join(mask_dir, mask_files[i])

    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)/255.0
    images.append(img)

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask.astype(np.float32)/255.0
    mask = np.expand_dims(mask, axis=-1)
    masks.append(mask)

X = np.array(images, dtype=np.float32)
y = np.array(masks, dtype=np.float32)

print("Loaded images:", X.shape)
print("Loaded masks:", y.shape)

# ----------------------
# BUILD U-NET MODEL
# ----------------------
def build_unet(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    b1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D()(b1)
    c3 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c3)
    c4 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c4)

    model = models.Model(inputs, outputs)
    return model

model = build_unet()

# ----------------------
# COMPILE MODEL
# ----------------------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------
# TRAIN MODEL
# ----------------------
if X.shape[0] == 0:
    print("No images found. Please check dataset folder!")
else:
    print("🚀 Training started...")
    model.fit(X, y, epochs=5, batch_size=4)

    # ----------------------
    # SAVE MODEL
    # ----------------------
    os.makedirs("models", exist_ok=True)
    model.save("models/unet_model.h5")
    print("✅ Model trained and saved successfully!")