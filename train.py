
import re
import pickle
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. Settings (Matching your notebook exactly)
vocab_size = 10000
max_length = 300
embedding_dim = 128
oov_tok = "<OOV>"

print("Loading Stanford NLP IMDB dataset from Hugging Face...")
ds = load_dataset("stanfordnlp/imdb")

# 2. Text Cleaning Function
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)      # Delete HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # Delete punctuation and numbers
    text = text.lower().strip()               # Uniform lowercase
    return text

print("Cleaning text data...")
train_texts = [clean_text(text) for text in ds['train']['text']]
test_texts = [clean_text(text) for text in ds['test']['text']]

train_labels = ds['train']['label']
test_labels = ds['test']['label']

print("Fitting Tokenizer on training text...")
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_texts)

# Save Tokenizer for your Streamlit UI web app
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved tokenizer.pickle locally.")

print(" Transforming text strings into padded sequences...")
X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_length, padding='post', truncating='pre')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_length, padding='post', truncating='pre')

y_train = np.array(train_labels)
y_test = np.array(test_labels)

print(" Building Deeper Hierirectional GRU Architecture...")
model = Sequential([
    Input(shape=(max_length,)),
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
    Dropout(0.4), 
    
    # Layer 1: Captures basic phrase contexts
    Bidirectional(GRU(units=64, dropout=0.2, return_sequences=True)), 
    
    # Layer 2: Captures abstract interactions like double negatives
    Bidirectional(GRU(units=32, dropout=0.2)), 
    
    Dense(units=32, activation='relu'),
    Dropout(0.4),
    Dense(units=1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

model.summary()

# Early stopping protection
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Starting local model training fit loop...")
history = model.fit(
    X_train, y_train,
    epochs=15,          
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Export the trained neural network
model.save('sentiment_gru_model.h5')
print("Complete! Model saved locally as 'sentiment_gru_model.h5'")