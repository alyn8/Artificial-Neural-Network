# 🧠 MNIST Classification with Artificial Neural Networks (ANN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alyn8/ANN/blob/main/ArtiFinalProject.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2F3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-orange)
![Keras](https://img.shields.io/badge/Keras-3.5+-red)
![License](https://img.shields.io/badge/License-MIT-green)

This project is an Artificial Neural Network (ANN) implementation designed to classify handwritten digits from the MNIST dataset using Deep Learning techniques. It covers model optimization, dynamic learning rate scheduling (Learning Rate Decay), and performance evaluation.

---

## 📌 Project Overview

- **Dataset:** MNIST Handwritten Digits Dataset (28x28 grayscale images)
- **Problem Type:** Multi-Class Classification (10 classes, digits 0–9)
- **Model Architecture:** Two-Hidden-Layer Fully Connected Artificial Neural Network (ANN)
- **Model Performance:** ~97.9% Test Accuracy

---

## 📐 Data Preprocessing & Model Architecture

### 1. Data Preprocessing
- **Normalization:** Pixel values are scaled from `[0, 255]` to `[0, 1]` (`x / 255.0`).
- **Reshaping:** 28x28 image matrices are flattened into 784-dimensional 1D vectors.
- **One-Hot Encoding:** Target labels (`y_train`, `y_test`) are converted into 10-element categorical vectors using Keras's `to_categorical`.

### 2. Model Architecture
The network includes **Dropout** layers to prevent overfitting during training:

| Layer | Neurons / Type | Activation / Parameters |
| :--- | :--- | :--- |
| **Input Layer** | 784 Nodes | Unrolled MNIST Vector |
| **Hidden Layer 1** | 64 Neurons | ReLU, `kernel_initializer='uniform'` |
| **Dropout** | Rate: 10% (0.1) | Overfitting Prevention |
| **Hidden Layer 2** | 64 Neurons | ReLU, `kernel_initializer='uniform'` |
| **Output Layer** | 10 Neurons | Softmax (Class Probabilities) |

---

## 🛠 Optimization & Learning Rate Strategies

The model utilizes the **Stochastic Gradient Descent (SGD)** optimizer alongside dynamic learning rate techniques:

1. **SGD Momentum + Time-based Decay:**
   - Initial Learning Rate ($lr_0$): `0.1`
   - Momentum: `0.8`
   - Decay Rate: $lr_0 / \text{Epochs}$
2. **Exponential Decay (LearningRateScheduler):**
   - Implemented via Keras `LearningRateScheduler` callback to exponentially reduce learning rate after each epoch:
     $$\text{lr} = \text{lr}_0 \times e^{(-\text{decay} \times \text{epoch})}$$

---

## 📊 Training & Evaluation Results

The model was trained for **60 Epochs** with batch sizes of **196 / 64**:

- **Training Accuracy:** ~98.9%
- **Validation Accuracy:** **~97.9%**
- **Validation Loss:** ~0.072

---

## 🔍 Code Structure & Workflow

The architecture and workflow of the code within the notebook are structured as follows:

```text
ArtiFinalProject.ipynb
│
├── 📦 1. Data Loading & Dependencies
│   ├── Importing Keras, TensorFlow, NumPy, Matplotlib
│   └── Loading train/test split via mnist.load_data()
│
├── 🧹 2. Data Preprocessing
│   ├── Flattening pixel matrices (Reshape: 28x28 ➔ 784)
│   ├── Data type conversion (float32) & Normalization (/ 255.0)
│   └── One-Hot Encoding target labels (to_categorical)
│
├── 🏗️ 3. ANN Model Architecture
│   ├── Defining Sequential model
│   ├── Stacking Dense (64, relu) + Dropout(0.1) layers
│   └── Defining Output layer (Dense 10, softmax)
│
├── ⚙️ 4. Hyperparameters & Optimization
│   ├── Configuring SGD Optimizer (Learning Rate & Momentum)
│   ├── Writing Learning Rate Scheduler / Decay functions
│   └── Model compilation (loss='categorical_crossentropy', metrics=['accuracy'])
│
├── 🚀 5. Model Training
│   ├── Running model.fit() (60 Epochs, Batch Size tuning)
│   └── Tracking training vs. validation loss/accuracy
│
└── 📈 6. Evaluation & Visualization
    ├── Evaluating performance on test dataset
    └── Plotting Training vs. Validation Loss and Accuracy curves

REQUIREMENTS:

Ensure you have the following dependencies installed in your environment:

-python >= 3.10

-tensorflow >= 2.17.0

-keras >= 3.5.0

-scikit-learn

-scikeras

-matplotlib

-numpy

   

📜 License

This project is open-source and available under the MIT License.
