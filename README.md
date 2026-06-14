# Sentify AI: Deep Hierarchical Sentiment Analysis Dashboard 🚀

An end-to-end Natural Language Processing (NLP) application that predicts text sentiment using a deep, stacked Bidirectional Gated Recurrent Unit (GRU) network. Built with TensorFlow/Keras and served via an interactive Streamlit web dashboard.

## 🧠 Model Architecture & Features
* **Data Source:** Stanford NLP IMDB Dataset (via Hugging Face).
* **Text Preprocessing:** Custom regex pipeline removing HTML artifacts, punctuation, and optimizing text casing.
* **Embedding Layer:** 128-dimensional word vectorization with zero-masking enabled to ignore padding sequences.
* **Recurrent Layers:** Stacked Hierarchical Bidirectional GRUs (Layer 1: 64 units for basic phrase contexts | Layer 2: 32 units for complex semantic interactions like double negatives).
* **Regularization:** Rigid Spatial and Dense Dropout boundaries (0.4) to combat overfitting.

## 📂 Repository Structure
* `app.py`: Streamlit front-end dashboard interface.
* `train.py`: Data loading, tokenization mapping, and deep learning neural network training loop.
* `preprocess.py`: Centralized text cleaning helper script.
* `result.py`: Independent evaluation engine calculating Precision, Recall, F1-score, and Confusion Matrices.
* `requirements.txt`: Environment configuration and exact package dependencies.

## 🛠️ How To Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Vishal-058/sentify-ai.git](https://github.com/Vishal-058/sentify-ai.git)
   cd sentify-ai
###Install dependencies:
	pip install -r requirements.txt

### Train the network
	python train.py

### launch the web dashboard
	streamlit run app.py


