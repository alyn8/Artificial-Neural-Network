# ARTIFICIAL NEURAL NETWORKS
# 🧠 MNIST ile Yapay Sinir Ağları (ANN) Sınıflandırma Projesi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alyn8/ANN/blob/main/ArtiFinalProject.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2F3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-orange)
![Keras](https://img.shields.io/badge/Keras-3.5+-red)
![License](https://img.shields.io/badge/License-MIT-green)

Bu proje, Derin Öğrenme (Deep Learning) yöntemlerini kullanarak el yazısı rakamların (MNIST Veri Seti) sınıflandırılmasını amaçlayan bir Yapay Sinir Ağı (Artificial Neural Network - ANN) uygulamasıdır. Proje kapsamında model optimizasyonu, öğrenme oranı değişimi (Learning Rate Decay/Scheduler) ve başarım analizi gerçekleştirilmiştir.

---

## 📌 Proje Özeti

- **Veri Seti:** MNIST El Yazısı Rakamlar Veri Seti (28x28 gri tonlamalı görüntüler)
- **Problem Türü:** Çok Sınıflı Sınıflandırma (0-9 arası rakamlar, toplam 10 sınıf)
- **Model Mimarisi:** İki Gizli Katmanlı Tam Bağlantılı Yapay Sinir Ağı (Fully Connected ANN)
- **Model Performansı:** ~%97.9 Doğruluk (Test Accuracy)

---

## 📐 Veri Ön İşleme & Model Mimarisi

### 1. Veri Ön İşleme (Data Preprocessing)
- **Normalizasyon:** Görüntü piksel değerleri `[0, 255]` aralığından `[0, 1]` aralığına çekilmiştir (`x / 255.0`).
- **Reshaping:** 28x28 boyutundaki piksel matrisleri 784 boyutlu tek boyutlu vektörlere dönüştürülmüştür.
- **One-Hot Encoding:** Etiketler (`y_train`, `y_test`), Keras'ın `to_categorical` fonksiyonu ile 10 elemanlı kategorik vektörlere dönüştürülmüştür.

### 2. Model Mimarisi
Model, aşırı öğrenmeyi (Overfitting) önlemek amacıyla **Dropout** katmanları ile desteklenmiştir:

| Katman (Layer) | Hücre Sayısı / Türü | Aktivasyon / Parametreler |
| :--- | :--- | :--- |
| **Giriş Katmanı (Input)** | 784 Düğüm | Unrolled MNIST vektörü |
| **1. Gizli Katman** | 64 Nöron | ReLU, `kernel_initializer='uniform'` |
| **Dropout** | Oran: %10 (0.1) | Overfitting önleme |
| **2. Gizli Katman** | 64 Nöron | ReLU, `kernel_initializer='uniform'` |
| **Çıkış Katmanı (Output)** | 10 Nöron | Softmax (Sınıf Olasılıkları) |

---

## 🛠 Optimizasyon ve Öğrenme Oranı Stratejileri

Model eğitiminde **SGD (Stochastic Gradient Descent)** optimizasyon algoritması ve farklı dinamik öğrenme oranı yaklaşımları test edilmiştir:

1. **SGD Momentum + Time-based Decay:**
   - İlk Öğrenme Oranı ($lr_0$): `0.1`
   - Momentum: `0.8`
   - Decay Rate: $lr_0 / \text{Epochs}$
2. **Exponential Decay (LearningRateScheduler):**
   - Keras `LearningRateScheduler` callback yapısı kullanılarak her epoch sonunda öğrenme oranı üstel olarak azaltılmıştır:
     $$\text{lr} = \text{lr}_0 \times e^{(-\text{decay} \times \text{epoch})}$$

---

## 📊 Eğitim ve Başarım Sonuçları

Model **60 Epoch** ve **196 / 64 Batch Size** seçenekleriyle eğitilmiştir:

- **Eğitim Doğruluğu (Train Accuracy):** ~%98.9
- **Doğrulama Doğruluğu (Validation Accuracy):** **~%97.9**
- **Doğrulama Kaybı (Validation Loss):** ~0.072

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.10 veya 3.11
- TensorFlow 2.17+
- Keras 3.5+
- Scikit-Learn
- Matplotlib
- NumPy

### Çalıştırma Adımları

1. Repoyu klonlayın:
   ```bash
   git clone [https://github.com/alyn8/ANN.git](https://github.com/alyn8/ANN.git)
   cd ANN
