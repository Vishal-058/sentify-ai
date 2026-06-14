
import re
import pickle
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

max_length = 300

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

print("Loading saved model weights and custom tokenizer...")
try:
    model = tf.keras.models.load_model('sentiment_gru_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Model and Tokenizer successfully loaded!")
except IOError as e:
    print(f"Initialization Error: Make sure your files exist. Details: {e}")
    exit()

print("Fetching test dataset split from Hugging Face...")
ds = load_dataset("stanfordnlp/imdb")
test_texts = [clean_text(text) for text in ds['test']['text']]
y_test = np.array(ds['test']['label'])

print("Vectorizing test sequences...")
X_test_seq = tokenizer.texts_to_sequences(test_texts)
X_test = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='pre')

print("Evaluating model accuracy...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nComputing prediction probabilities...")
y_pred_probs = model.predict(X_test, batch_size=64, verbose=1)
y_pred = (y_pred_probs > 0.5).astype("int32")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))