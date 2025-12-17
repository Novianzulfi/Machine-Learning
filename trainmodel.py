# ===============================
# TRAIN MODEL DAUN TERONG
# Image Classification - 7 Classes
# ===============================

import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# 1. PATH DATASET & PARAMETER
# -------------------------------
DATASET_DIR = "dataset"   # folder dataset utama
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# 2. DATA GENERATOR (PREPROCESSING)
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

with open("model/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f, indent=4)

print("Class indices:", train_generator.class_indices)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes

print("\nJumlah kelas:", NUM_CLASSES)
print("Nama kelas:", train_generator.class_indices)

# -------------------------------
# 3. SIMPAN CLASS INDICES (PENTING)
# -------------------------------
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_generator.class_indices, f, indent=4)

# -------------------------------
# 4. BUILD MODEL (TRANSFER LEARNING)
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------------
# 5. COMPILE MODEL
# -------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 6. CALLBACK (EARLY STOPPING)
# -------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -------------------------------
# 7. TRAIN MODEL
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# -------------------------------
# 8. SIMPAN MODEL
# -------------------------------
model_path = os.path.join(MODEL_DIR, "model_daun_terong.h5")
model.save(model_path)

print("\n✅ Training selesai!")
print("✅ Model disimpan di:", model_path)
print("✅ Class indices disimpan di:", os.path.join(MODEL_DIR, "class_indices.json"))
