# brain-tumor-detection-cnn
# Brain Tumor Detection using CNN on MRI images (Keras &amp; TensorFlow)

## Overview

This project implements a Convolutional Neural Network (CNN) for detecting brain tumors from MRI images, based on the research paper:

**"Brain Tumor Detection using MRI Images and Convolutional Neural Network"**

The model classifies MRI scans into:

* Tumor
* No Tumor

---

## Objectives

* Apply deep learning to medical imaging
* Build a CNN model from scratch
* Achieve high accuracy in tumor detection
* Reproduce results from the research paper (~96% accuracy)

---

## Dataset

The dataset is publicly available on Kaggle:

🔗 https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri

It contains MRI images categorized into:

* `yes` → tumor
* `no` → no tumor

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy & Pandas
* Matplotlib & Seaborn
* Scikit-learn

---

## Model Architecture

* 3 Convolutional Layers
* MaxPooling + BatchNormalization
* Fully Connected Layers (Dense)
* Dropout for regularization
* Softmax output (2 classes)

---

## Results

* Accuracy: ~96%
* Loss and Accuracy curves plotted
* Confusion Matrix included

---

## Sample Output

(Add sample prediction images here if you want)

---

## Future Improvements

* Use Transfer Learning (ResNet, EfficientNet)
* Hyperparameter tuning
* Data augmentation
* 3D MRI analysis

---

## Author

Zohreh Eram

---

## ⭐ If you like this project

Give it a star on GitHub!
