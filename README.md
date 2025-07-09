# 🧠 Brain Tumor Detection Using CNN

Detect brain tumors from MRI images using a custom Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## 🌟 Overview

This project leverages deep learning to automatically classify brain MRI images as **Tumor** or **No Tumor**. The model is trained on a labeled dataset of MRI scans and achieves high accuracy using a carefully designed CNN architecture.

---

## 🖼️ Sample Images

| Tumor (Yes) Example | No Tumor Example |
|:-------------------:|:----------------:|
| ![Tumor](assets/yes_sample.jpg) | ![No Tumor](assets/no_sample.jpg) |

> _Replace the above image links with your own sample images from the dataset for best results!_

---

## 🚀 Tech Stack

- **Python 3**
- **TensorFlow & Keras**: Deep learning framework for model building and training
- **OpenCV**: Image processing and loading
- **NumPy**: Numerical operations
- **scikit-learn**: Data splitting and evaluation metrics

---

## 🧩 Model Architecture

The CNN model consists of the following layers:

| Layer Type         | Output Shape      | Parameters |
|--------------------|------------------|------------|
| Input (224x224x3)  | (224, 224, 3)    | 0          |
| Conv2D (32 filters, 3x3) | (222, 222, 32) | 896        |
| MaxPooling2D       | (111, 111, 32)   | 0          |
| Conv2D (64 filters, 3x3) | (109, 109, 64) | 18496      |
| MaxPooling2D       | (54, 54, 64)     | 0          |
| Dropout (0.25)     | (54, 54, 64)     | 0          |
| Flatten            | (186624)         | 0          |
| Dense (64 neurons) | (64)             | 11944000   |
| Dropout (0.5)      | (64)             | 0          |
| Dense (1 neuron, sigmoid) | (1)      | 65         |

**Total Parameters:** ~12 million  
**Activation Functions:** ReLU (hidden layers), Sigmoid (output)

---

## 📊 Training & Evaluation

- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Callbacks:** EarlyStopping, TensorBoard

After training, the model is evaluated using classification report and confusion matrix for detailed performance analysis.

---

## 🏁 How to Run

1. **Clone the repository and install dependencies:**
    ```sh
    pip install tensorflow opencv-python scikit-learn numpy
    ```

2. **Prepare your dataset:**
    - Place MRI images in `brain_tumor_dataset/yes/` and `brain_tumor_dataset/no/` folders.

3. **Train the model:**
    ```sh
    python cnnModel.py
    ```

4. **Test on new images:**
    ```sh
    python texstingmodel.py
    ```

---

## 🧪 Example Prediction

```python
Result: Tumor Detected
Confidence: 0.98
```

---

## 📈 Results

- **Test Accuracy:** _See console output after training_
- **Classification Report & Confusion Matrix:** Printed after training

---

## 🛠️ Files

- [`cnnModel.py`](cnnModel.py): Model training and evaluation
- [`texstingmodel.py`](texstingmodel.py): Load trained model and predict on new images
- `brain_tumor_model.h5`: Saved trained model

---

## 👨‍💻 Author

- **Your Name**  
  _Add your contact or GitHub profile here!_

---

## 📷 Tools & Libraries Used

| Tool/Library | Logo |
|--------------|------|
| TensorFlow   | ![TensorFlow](https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg) |
| Keras        | ![Keras](https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg) |
| OpenCV       | ![OpenCV](https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg) |
| NumPy        | ![NumPy](https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg) |
| scikit-learn | ![scikit-learn](https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg) |

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [OpenCV Documentation](https://opencv.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

## 📂 Dataset

This project uses the [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?resource=download) dataset from Kaggle.  
It contains MRI images categorized into `yes` (tumor) and `no` (no tumor) folders, which are used for training and evaluating the model.

---

> _This project is for educational purposes. For medical use, always consult a professional._
