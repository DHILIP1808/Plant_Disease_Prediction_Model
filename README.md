
# 🌿 Plant Disease Classification using EfficientNetB2 

This project leverages the power of EfficientNetB2 for accurate classification of plant leaf diseases. The goal is to help farmers and researchers identify diseases early using AI-based computer vision techniques.

## 📁 Project Structure

```bash
.
├── Model_Training.ipynb     # Notebook for training ViT model on plant disease dataset
├── Model_Testing.ipynb      # Notebook to test trained model on test images
├── Model_Prediction.ipynb   # Notebook for single image prediction
└── README.md                # Project documentation
```

## 📌 Dataset

We use the **New Plant Diseases Dataset** from Kaggle, which includes healthy and diseased leaf images of various crops like tomato, maize, grape, potato, etc.

📎 **[Dataset Link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)**

## 🧠 Model Overview

We use the **EfficientNetB2** model from HuggingFace Transformers for fine-tuning on the plant disease dataset.

Key Features:
- Pretrained ViT model
- Fine-tuned on leaf disease dataset
- Torch + HuggingFace integration
- Modular notebooks for training, testing, and prediction

## 🚀 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/plant-disease-vit.git
cd plant-disease-vit
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install torch torchvision transformers evaluate
```

### 3. Download Dataset

Download and extract the dataset from the Kaggle link above and structure it like this:

```bash
dataset/
├── train/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   └── ...
├── valid/
│   └── ...
└── test/
    └── (only images, no subfolders)
```

## 📓 Notebooks

- **`Model_Training.ipynb`**  
  Fine-tunes the pretrained ViT model using the training and validation sets.

- **`Model_Testing.ipynb`**  
  Loads the trained model and evaluates it on test images (flat folder, ~33 images).

- **`Model_Prediction.ipynb`**  
  Makes predictions on a single image with model confidence scores.

## 🧪 Sample Results

✅ Accurate classification of tomato, grape, maize, and potato leaf diseases  
📊 Precision, recall, and F1-score evaluation with `evaluate` library  
🖼️ Image prediction visualization with confidence values

## 📷 Example Prediction

```python
# Example: Predicting a test image
>>> predict_image("test/Tomato_blight_example.jpg")
Predicted Class: Tomato___Late_blight (Confidence: 98.2%)
```

## 📚 References
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## ✨ Future Work

- Add image similarity using CLIP + FAISS
- Deploy as a web app using FastAPI or Streamlit
- Integrate LLM-based recommendations

## 🤝 Contributing

Pull requests and suggestions are welcome! For major changes, please open an issue first.

## 📜 License

This project is licensed under the MIT License.
