# -*- coding: utf-8 -*-
"""ulnn_training

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ONnZ5OCP1_fpdeZ6GK9e7T4BYnsOPkOX
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from zipfile import ZipFile

# Step 1: Upload and Extract Zip
data_path = '/content/reduced_dataset_zipfile.zip'

with ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall()
    print('The dataset has been extracted.')

# Step 2: Set image size and paths
IMG_SIZE = 256
BATCH_SIZE = 32
DATASET_PATH = '/content/Atmospheric_condition'

# Step 3: Simple image loader without augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Step 4: CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Step 5: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Show model summary
model.summary()

# Step 7: Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 8: Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# Step 9: Save the model
model.save("weather_classification_model.h5")
print("Model saved as weather_classification_model.h5")

# Save the model
model.save("weather_classification_model.h5")

from google.colab import files
files.download('weather_classification_model.h5')

import numpy as np
import cv2
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("weather_classification_model.h5")

# Define correct class order based on your training
classes = ['fogsmog', 'lightning', 'rain' , 'sandstorm' , 'snow']

# Test image path
image_path = "/content/fogsmoke_images.jpg"  # CHANGE THIS for each test

# Preprocess the image
IMG_SIZE = 256
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0  # Only if you used rescale=1./255 during training
img = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img)
print("Raw prediction scores:", predictions)
print("Predicted class:", classes[np.argmax(predictions)])

from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# Step 10: Predict on the validation set
val_generator.reset()  # Reset generator before prediction
pred_probs = model.predict(val_generator, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)  # Convert one-hot to class index

# Get true labels
y_true = val_generator.classes

# Get class labels (index to class name mapping)
class_labels = list(val_generator.class_indices.keys())

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)

# Print classification report
print(report_df)

# Optional: Save to CSV
report_df.to_csv("weather_classification_report.csv")