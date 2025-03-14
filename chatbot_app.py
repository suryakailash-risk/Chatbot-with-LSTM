import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
MODEL_PATH = "chatbot_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN_PATH = "max_len.txt"

# Load the saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load max_len
with open(MAX_LEN_PATH, "r") as f:
    max_len = int(f.read())

# Chatbot response function
def predict_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    predicted_probs = model.predict(input_seq, verbose=0)[0]

    predicted_indices = np.argmax(predicted_probs, axis=-1)
    predicted_words = [tokenizer.index_word[idx] for idx in predicted_indices if idx in tokenizer.index_word]

    return " ".join(predicted_words)

# Streamlit UI
st.title("🤖 Chatbot - AI Conversational Assistant")
st.write("Type your message below to chat with the AI.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.text_input("You:", "")

if user_input:
    response = predict_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for speaker, text in st.session_state.chat_history:
    st.write(f"**{speaker}:** {text}")

