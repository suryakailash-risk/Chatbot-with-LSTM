import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import pickle  # For saving tokenizer

# Load dataset
DATASET_PATH = "Conversation.csv"
conversation_data = pd.read_csv(DATASET_PATH)

# Ensure no NaN values and convert all to strings
conversation_data['question'] = conversation_data['question'].fillna("").astype(str)
conversation_data['answer'] = conversation_data['answer'].fillna("").astype(str)

# Prepare input and target text
X_texts, y_texts = list(conversation_data['question']), list(conversation_data['answer'])

# Tokenizer for text processing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_texts + y_texts)
total_words = len(tokenizer.word_index) + 1

# Convert texts to sequences
X_sequences = tokenizer.texts_to_sequences(X_texts)
y_sequences = tokenizer.texts_to_sequences(y_texts)

# Padding sequences for uniform length
max_len = max(max(len(seq) for seq in X_sequences), max(len(seq) for seq in y_sequences))
X_sequences = pad_sequences(X_sequences, maxlen=max_len, padding='post')
y_sequences = pad_sequences(y_sequences, maxlen=max_len, padding='post')

# Convert target (y) to one-hot encoding
y_sequences = to_categorical(y_sequences, num_classes=total_words)

# Define the LSTM Model
model = Sequential([
    Embedding(input_dim=total_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train and save the model
model.fit(X_sequences, y_sequences, epochs=500, verbose=2)

# Save the model
model.save("chatbot_model.h5")

# Save the tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save max sequence length
with open("max_len.txt", "w") as f:
    f.write(str(max_len))

print("Model and tokenizer saved successfully!")
