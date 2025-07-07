"""
Edge AI Prototype: Recyclable Item Classifier using TensorFlow Lite
Author: [Your Name]
"""

# ========== Step 1: Import Libraries ==========
import os
import zipfile
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ========== Step 2: Download and Extract Dataset (4 classes) ==========
DATASET_URL = "https://github.com/rahelmekonnen/recyclable-classifier-dataset/raw/main/recyclables_small.zip"
DATASET_PATH = "recyclables"

if not os.path.exists(DATASET_PATH):
    print("🔽 Downloading sample dataset...")
    response = requests.get(DATASET_URL)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall()
    print("✅ Dataset downloaded and extracted.")
else:
    print("✅ Dataset already exists.")

# ========== Step 3: Set Parameters ==========
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

# ========== Step 4: Load Dataset ==========
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ========== Step 5: Build Lightweight CNN ==========
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ========== Step 6: Train Model ==========
print("\n📚 Training model...")
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# ========== Step 7: Evaluate Model ==========
loss, acc = model.evaluate(val_data)
print(f"\n✅ Validation Accuracy: {acc:.2%}")

# ========== Step 8: Convert to TensorFlow Lite ==========
print("🔄 Converting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ model.tflite saved.")

# ========== Step 9: Simulate TFLite Inference ==========
def preprocess_image(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)

def predict_tflite(image_path, interpreter, input_details, output_details, class_names):
    input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output)
    confidence = np.max(output)
    return class_names[predicted_index], confidence

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names
class_names = list(train_data.class_indices.keys())

# Pick one image for test inference
sample_path = os.path.join(DATASET_PATH, class_names[0], os.listdir(os.path.join(DATASET_PATH, class_names[0]))[0])
label, confidence = predict_tflite(sample_path, interpreter, input_details, output_details, class_names)
print(f"🔍 Sample Inference Result → Predicted: {label} ({confidence*100:.2f}%)")
