# AI-Driven Crack Detection System 🏗️🧠

## 📌 Project Overview
An AI-powered computer vision system that detects cracks in structural images
using a Convolutional Neural Network (CNN). The model classifies images as
**Crack** or **No Crack** to assist in automated infrastructure inspection.

## 🎯 Purpose
- Automate crack detection in buildings and roads
- Reduce manual inspection cost and time
- Improve structural safety monitoring

## 🧠 Technologies Used
- Python
- TensorFlow & Keras
- OpenCV
- NumPy, Matplotlib
- CNN (Deep Learning)

## 🖼 Dataset
- Image dataset containing cracked and non-cracked surfaces
- Images resized and augmented for better generalization

## ⚙️ Model Architecture
- Convolutional Neural Network (CNN)
- Binary Classification (Crack / No Crack)
- Activation: ReLU, Sigmoid
- Loss Function: Binary Crossentropy
- Optimizer: Adam

## 📊 Results
- Training Accuracy: ~XX%
- Validation Accuracy: ~XX%
- Visualized loss & accuracy curves included

## 🚀 How to Run

```bash
git clone https://github.com/your-username/ai-driven-crack-detection-system.git
cd ai-driven-crack-detection-system
pip install -r requirements.txt
python src/predict.py
