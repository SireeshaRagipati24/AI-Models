
# 🐱🐶 Cat vs Dog Image Classification — CNN with Deep Learning

> **Teaching a neural network to see** — building a Convolutional Neural Network from scratch that classifies images as cats or dogs with 80%+ accuracy.

---

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Domain** | Computer Vision / Image Classification |
| **Architecture** | Convolutional Neural Network (CNN) |
| **Framework** | TensorFlow / Keras |
| **Training Images** | 8,048 (cats + dogs) |
| **Test Images** | 2,000 (cats + dogs) |
| **Input Shape** | 64 × 64 × 3 (RGB) |
| **Task** | Binary Classification — Cat (0) or Dog (1) |
| **Target Accuracy** | 80%+ |

---

## 🎯 Business Objective

> Build a deep learning model that can **automatically identify whether an image contains a cat or a dog** — demonstrating end-to-end computer vision pipeline from raw images to production-ready predictions.

**Real-World Applications of Image Classification:**
- 🐾 **Pet adoption platforms** — auto-tag and categorise animal photos
- 🏥 **Medical imaging** — classify X-rays, MRI scans, skin lesions
- 🚗 **Autonomous vehicles** — detect pedestrians, road signs, obstacles
- 🛡️ **Security systems** — facial recognition, anomaly detection
- 🛍️ **E-commerce** — visual product search (find similar items)

---

## 🧠 Why CNN for Image Classification?

Traditional ML models fail at raw pixel data — a 64×64 RGB image = **12,288 features**, and they lose all spatial relationships.

| Approach | Limitation |
|----------|-----------|
| Random Forest / SVM | Treats pixels independently — misses spatial structure |
| ANN (Flat) | Flattens image → loses "where" features are |
| **CNN** | ✅ Learns spatial hierarchy — edges → shapes → objects |

**CNNs learn features automatically:**
```
Conv Layer 1 → Detects: edges, colour gradients
Conv Layer 2 → Detects: shapes (curves, circles, lines)
Conv Layer 3 → Detects: high-level patterns (eyes, ears, fur)
Dense Layer  → Combines patterns → "This is a cat"
```

---

## 🏗️ CNN Architecture

```
Input Image (64×64×3 RGB)
         ↓
┌─────────────────────────────────────────┐
│  Conv2D (32 filters, 3×3, ReLU)         │  ← Detect low-level features
│  MaxPooling2D (2×2) → 32×32            │  ← Downsample
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Conv2D (64 filters, 3×3, ReLU)         │  ← Detect mid-level features
│  MaxPooling2D (2×2) → 16×16            │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Conv2D (128 filters, 3×3, ReLU)        │  ← Detect high-level features
│  MaxPooling2D (2×2) → 8×8              │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Flatten → 8×8×128 = 8,192 features    │
│  Dense (128 units, ReLU)               │
│  Dropout (0.5)  ← Prevents overfitting │
│  Dense (1 unit, Sigmoid)               │
└─────────────────────────────────────────┘
         ↓
  Output: P(Dog) ∈ [0, 1]
  < 0.5 = Cat 🐱  |  ≥ 0.5 = Dog 🐶
```

---

## 🗺️ Project Workflow

```
Load Dataset (8,048 train + 2,000 test images)
               ↓
Data Augmentation (shear, zoom, flip, rotate, shift)
               ↓
Build 3-Layer CNN Architecture
               ↓
Compile (Adam + Binary Crossentropy)
               ↓
Train with Smart Callbacks (EarlyStopping + ReduceLROnPlateau)
               ↓
Plot Training History (Accuracy + Loss curves)
               ↓
Evaluate on Test Set (Accuracy, Confusion Matrix)
               ↓
Single Image Prediction (predict_image() function)
               ↓
Save Model → cat_dog_cnn.h5
```

---

## 📊 Data Augmentation

Applied to **training data only** — artificially increases diversity to prevent overfitting:

| Augmentation | Effect |
|-------------|--------|
| `rescale=1/255` | Normalize pixels 0–255 → 0–1 |
| `shear_range=0.2` | Slanted/tilted perspective |
| `zoom_range=0.2` | Random zoom in/out |
| `horizontal_flip=True` | Mirror images left-right |
| `rotation_range=15` | Rotate up to ±15° |
| `width_shift_range=0.1` | Shift image horizontally |
| `height_shift_range=0.1` | Shift image vertically |

> Augmentation makes the model see each image differently every epoch — the model cannot "memorise" the training set, forcing it to learn genuine features.

---

## 🔧 Smart Training Callbacks

| Callback | What It Does | Why Important |
|----------|-------------|---------------|
| **EarlyStopping** | Stops if val_loss doesn't improve for 5 epochs | Prevents overfitting & wasted compute |
| **ReduceLROnPlateau** | Halves learning rate when training plateaus | Helps converge to better solution |

---

## 📂 Folder Structure

```
CNN_cat_dog/
│
├── training_set/
│   ├── cats/       ← ~4,000 cat images
│   └── dogs/       ← ~4,000 dog images
├── test_set/
│   ├── cats/       ← ~1,000 cat images
│   └── dogs/       ← ~1,000 dog images
├── single_prediction/
│   ├── cat_or_dog_1.jpg
│   └── cat_or_dog_2.jpg
├── 📓 CNN_cat_dog_classification.ipynb
├── 🤖 cat_dog_cnn.h5     ← Saved trained model
└── 📝 README.md
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/SireeshaRagipati24/Machine-Learning.git
cd Machine-Learning/CNN

pip install tensorflow keras numpy pillow matplotlib scikit-learn

jupyter notebook CNN_cat_dog_classification.ipynb
```

---

## 🔮 Predict on a New Image

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load saved model
model = load_model('cat_dog_cnn.h5')

def predict_image(image_path):
    img = Image.open(image_path).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # Add batch dim
    
    result = model.predict(img_batch)[0][0]
    return 'Dog 🐶' if result >= 0.5 else 'Cat 🐱', result

label, prob = predict_image('your_image.jpg')
print(f'Prediction: {label} (confidence: {prob*100:.1f}%)')
```

---

## 💡 Key Deep Learning Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Convolution | 3 × Conv2D with 32/64/128 filters |
| Spatial Hierarchy | Edges → Shapes → Patterns (3 layers) |
| MaxPooling | 64×64 → 32×32 → 16×16 → 8×8 |
| Data Augmentation | 7 techniques — prevents overfitting |
| Dropout (0.5) | Regularisation — drops 50% neurons during training |
| Early Stopping | Auto-stops + restores best weights |
| Binary Crossentropy | Loss for 2-class probability output |
| Sigmoid Activation | Output ∈ [0,1] → interpretable probability |
| ImageDataGenerator | Memory-efficient batch loading |

---

## 🚀 Future Improvements

- [ ] **Transfer Learning** — use VGG16/ResNet50 pretrained on ImageNet → 95%+ accuracy
- [ ] Add **Batch Normalisation** after each Conv layer for faster training
- [ ] Build a **Streamlit web app** — drag & drop image → instant prediction
- [ ] Extend to **multi-class** — cats, dogs, birds, fish, rabbits
- [ ] Try **Grad-CAM** visualisation — see which parts of image the model focuses on

---

## 🙋‍♀️ About Me

**Sireesha Ragipati** — Aspiring Data Analyst | Deep Learning Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sireesha-ragipati-269a10244/)

---

*⭐ If you found this helpful, give it a star!*
