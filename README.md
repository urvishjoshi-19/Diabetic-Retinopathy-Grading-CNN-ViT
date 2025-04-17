# Diabetic Retinopathy Detection using Deep Learning

This project presents a deep learning-based approach to detect **Diabetic Retinopathy** (DR) from retinal images. Diabetic Retinopathy is a complication of diabetes that affects the eyes and can lead to blindness if not diagnosed early. Leveraging convolutional neural networks (CNNs), this model aims to classify the severity of DR into different stages based on fundus images.

## 🔍 Problem Statement

Early detection of Diabetic Retinopathy is crucial to prevent vision loss. Manual diagnosis by ophthalmologists is time-consuming and subjective. Automating the detection process using deep learning can assist in faster and more accurate screening of DR.

## 📂 Dataset

The model uses the **APTOS 2019 Blindness Detection Dataset**, which contains high-resolution retinal images labeled with DR severity:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

Images are preprocessed to improve quality and resized to standard dimensions for training.

## 🧠 Model Architecture

The model uses **EfficientNetB0**, a powerful and lightweight CNN architecture pretrained on ImageNet. Transfer learning is applied to fine-tune the network on DR classification.

### Key Features:
- EfficientNetB0 backbone
- GlobalAveragePooling
- Dense layers with Dropout regularization
- Adam optimizer and Categorical Crossentropy loss
- Metrics: Accuracy and F1 Score

## ⚙️ Project Workflow

1. **Data Preprocessing**:
   - Image resizing and enhancement
   - Label encoding
   - Train-validation split

2. **Model Building**:
   - Load EfficientNetB0
   - Add custom classifier layers
   - Compile and train the model

3. **Evaluation**:
   - Accuracy and F1 score on validation set
   - Confusion matrix for class-wise performance

## 📈 Results

The trained model achieves competitive accuracy and is capable of distinguishing between different stages of DR. Visualization techniques such as confusion matrix and classification report provide insights into model performance.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV, PIL
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
   cd diabetic-retinopathy-detection
