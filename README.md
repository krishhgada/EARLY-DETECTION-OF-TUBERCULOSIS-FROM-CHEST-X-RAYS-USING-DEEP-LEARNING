# Early Detection of Tuberculosis Using Deep Learning (ResNet50 + Image Preprocessing)
Overview

This project implements a deep learning–based system for early detection of Tuberculosis (TB) from chest X-ray (CXR) images using ResNet50 and advanced preprocessing techniques. With Contrast Limited Adaptive Histogram Equalization (CLAHE) and Gaussian noise regularization, the model achieves 99.28% accuracy and an AUC of 0.9989, outperforming most existing TB detection approaches.

Dataset

Source: Publicly available Kaggle TB Chest X-ray dataset

Size: 4,600 CXRs

TB-positive: 800 images

TB-negative: 3,800 images

Split: 70% Training, 20% Validation, 10% Test (stratified)

Image Size: Resized to 232×232 pixels for model compatibility

Preprocessing & Augmentation

1. CLAHE: Enhances local contrast to reveal subtle TB lesions
2. Gaussian Noise Injection: Improves robustness and prevents overfitting
3. Data Augmentation (via Keras):

Random rotations (±5°)

Horizontal flips

Brightness scaling (±10%)

Zoom (±5%)

Width/Height shifts (±5%)

Model Architecture

Base Model: ResNet50 (pre-trained on ImageNet, frozen weights)

Custom Head:

Global Average Pooling

Dense(512) + ReLU

Dropout(0.3)

Dense(1) + Sigmoid (Binary Classification)

Loss: Binary Cross-Entropy
Optimizer: Adam (lr = 0.001)
Callbacks: EarlyStopping, ReduceLROnPlateau

Results
Metric	Score
Accuracy	99.28%
Precision (TB)	100%
Recall (TB)	95.8%
AUC	0.9989

Confusion Matrix: 1 false negative, 0 false positives

ROC AUC: Almost perfect separation of TB vs non-TB cases

Future Work

External dataset validation for generalization

Fine-tuning top ResNet50 layers for further accuracy gains

Potential integration into real-world TB screening tools
