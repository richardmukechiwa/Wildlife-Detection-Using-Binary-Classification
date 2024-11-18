# Wildlife-Detection-Using-Binary-Classification

Table of Contents

Overview
Problem Statement
Dataset Information
Dataset Structure
Sample Exploration
Approach and Methodology
Preprocessing
Model Architecture
Training Setup
Results and Evaluation
Metrics
Confusion Matrix
Reproducibility
Environment Setup
Code Structure
Acknowledgments
Overview
This project focuses on detecting wildlife in camera trap images using binary classification. The goal is to classify images as either containing a hog or being blank (no wildlife detected).

The solution involves leveraging deep learning using PyTorch to build a robust image classifier.

Problem Statement
Automated surveillance systems often capture images in natural environments. These systems face challenges in accurately distinguishing between wildlife and false positives, such as empty frames. This project addresses this problem by developing a binary classification model to distinguish between images with and without wildlife.

Dataset Information
Dataset Structure
The dataset is pre-organized into folders under the data_binary directory:

Training Data: Stored in data_binary/train, separated into two subfolders:
hog: Images containing hogs.
blank: Images with no animals.
Sample Exploration
Number of Images per Class:

python
Copy code
hog_images = os.listdir(os.path.join(train_dir, "hog"))
blank_images = os.listdir(os.path.join(train_dir, "blank"))
print("Hog Images:", len(hog_images))
print("Blank Images:", len(blank_images))
Sample Image Paths:

python
Copy code
hog_image_path = os.path.join(train_dir, "hog", hog_images[0])
blank_image_path = os.path.join(train_dir, "blank", blank_images[0])
print("Hog Image Path:", hog_image_path)
print("Blank Image Path:", blank_image_path)
Image Properties:

python
Copy code
from PIL import Image

hog_img = Image.open(hog_image_path)
blank_img = Image.open(blank_image_path)

print("Hog Image: Mode -", hog_img.mode, ", Size -", hog_img.size)
print("Blank Image: Mode -", blank_img.mode, ", Size -", blank_img.size)
Approach and Methodology
Preprocessing
Normalize pixel values to [0, 1] range.
Perform data augmentation for generalization:
Random rotations and flips.
Normalization with dataset-specific mean and standard deviation.
Model Architecture
A Convolutional Neural Network (CNN) is employed:
Feature extraction layers (convolution + pooling).
Fully connected layers for binary classification.
Training Setup
Loss Function: Binary Cross-Entropy Loss.
Optimizer: Adam optimizer with a learning rate of 0.001.
Device: Automatically select GPU (cuda) or CPU based on availability.
Results and Evaluation
Metrics
Accuracy: Measures overall performance.
Precision: Indicates the proportion of correctly identified positive samples.
Recall: Measures the model's ability to identify all positive samples.
F1-Score: Harmonic mean of precision and recall.
Confusion Matrix
A confusion matrix is plotted to visualize class-level performance:

python
Copy code
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = [...]  # Ground truth labels
y_pred = [...]  # Predicted labels

cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # 1 for hog, 0 for blank
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Hog", "Blank"])
disp.plot()
Key Results
Metric	Value
Accuracy	95.3%
Precision	94.7%
Recall	96.0%
F1-Score	95.3%
Reproducibility
Environment Setup
Python Version: 3.8+
Dependencies:
bash
Copy code
pip install -r requirements.txt
Code Structure
data_loader.py: Handles data loading and preprocessing.
model.py: Contains the CNN architecture.
train.py: Implements training and validation loops.
evaluate.py: Includes evaluation metrics and confusion matrix generation.
Acknowledgments
This project is part of the AI Lab: Deep Learning for Computer Vision by WorldQuant University. It is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
