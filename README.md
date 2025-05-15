# Weather Classification using CNN

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images into different atmospheric weather conditions such as fog/smog, lightning, rain, sandstorm, and snow.

## 📁 Dataset

The dataset used must be provided as a zipped folder (`reduced_dataset_zipfile.zip`) containing categorized subfolders of images. It is expected to be structured like this:


The script handles the extraction and loading of the dataset automatically.

## 🧠 Model Architecture

- Convolutional layers: 3 (with increasing filters: 32, 64, 128)
- Pooling layers: MaxPooling after each Conv2D
- Dense layers: 1 hidden (128 units + dropout), 1 output (softmax)
- Activation functions: ReLU for hidden layers, Softmax for output
- Optimizer: Adam
- Loss: Categorical Crossentropy

## 🚀 Training Details

- Image size: 256x256
- Batch size: 32
- Train-validation split: 80/20
- Early stopping: Enabled (patience = 5)
- Epochs: 20

## 🧪 Evaluation

- Generates a classification report on the validation dataset using `sklearn.metrics.classification_report`
- Outputs precision, recall, f1-score per class
- Optionally exports report as CSV (`weather_classification_report.csv`)

## 📦 Output

- Trained model saved as: `weather_classification_model.h5`
- Classification report saved as: `weather_classification_report.csv`

## 🖼️ Image Prediction
 Requirements
Python 3.7+

TensorFlow

NumPy

Pandas

OpenCV

scikit-learn

Usage
Upload the zipped dataset (reduced_dataset_zipfile.zip).

Run the script: ulnn_training.py.

The model is trained, saved, and tested on the validation set.

A classification report is printed and saved.

You can test the model on custom images. Update the `image_path` variable with your test image path:

```python
image_path = "/content/fogsmoke_images.jpg"
