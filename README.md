
# 🤖 AI Chatbot using TensorFlow & Streamlit  

This project is an **AI-powered chatbot** designed to interact and respond intelligently to user queries using **Deep Learning**. It utilizes **LSTM-based neural networks**, **Natural Language Processing (NLP)**, and **Streamlit** for an interactive web interface.  

## 🚀 Features  
✅ **LSTM-based Chatbot Model** – Uses a **Bidirectional LSTM** for better contextual understanding.  
✅ **Custom Trained on Conversations Dataset** – Ensures meaningful responses.  
✅ **Efficient Text Tokenization & Sequence Processing** – Prepares input for accurate output generation.  
✅ **Deployed Using Streamlit** – Provides a user-friendly chat interface.  

---

## 🛠 Tech Stack  
- **Programming Language**: Python 🐍  
- **Machine Learning Framework**: TensorFlow, Keras  
- **Web Deployment**: Streamlit  
- **Data Processing**: Pandas, NumPy  
- **Tokenizer & Text Sequences**: TensorFlow’s Text Preprocessing  

---

## 📂 Project Structure  
```bash
📁 AI-Chatbot-Project
│── 📄 train_chatbot.py       # Train and save chatbot model  
│── 📄 chatbot_app.py         # Streamlit-based chatbot app  
│── 📄 Conversation.csv       # Dataset with questions & answers  
│── 📄 chatbot_model.h5       # Trained chatbot model  
│── 📄 tokenizer.pkl          # Tokenizer for text processing  
│── 📄 max_len.txt            # Maximum sequence length  
│── 📄 README.md              # Project documentation  
```

---

## 🏗 How to Set Up & Run  

### 1️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Chatbot  
Run the script to train and save the chatbot model:  
```bash
python train_chatbot.py
```

### 3️⃣ Run the Chatbot Web App  
After training, launch the chatbot using Streamlit:  
```bash
streamlit run chatbot_app.py
```

---

## 🧠 How It Works  
1. **Train the Model** – The chatbot learns from the dataset (`Conversation.csv`) using **LSTM layers**. (https://www.kaggle.com/datasets/kreeshrajani/3k-conversations-dataset-for-chatbot) 
2. **Generate Responses** – User input is tokenized, processed, and passed through the model for prediction.  
3. **Deploy the Chatbot** – A web interface is provided for real-time chat interactions.  

---

## 📈 Model Details  
- **Embedding Layer** – Converts words into dense vector representations.  
- **Bidirectional LSTM Layers** – Enhances context understanding by processing data both forward and backward.  
- **Dropout Layers** – Prevents overfitting during training.  
- **Dense Layers** – Predicts the best response from the learned dataset.  

---

## 🚀 Future Improvements  
- Integrate **transformers (GPT-based models)** for more advanced responses.  
- Enable chatbot **memory retention** for longer conversations.  
- Support **multi-language responses** using NLP techniques.  

---

## 💡 Contributing  
Want to improve this chatbot? Contributions are welcome!  
1. Fork this repository.  
2. Create a new branch: `git checkout -b feature-branch`  
3. Commit your changes: `git commit -m "Add new feature"`  
4. Push to the branch: `git push origin feature-branch`  
5. Submit a Pull Request!  

---

## 📩 Contact  
For questions or suggestions, feel free to reach out!  

📧 **Email:** sramesh@seattleu.edu  
🔗 **LinkedIn:** [Profile](https://linkedin.com/in/suryakailash)  

⭐ If you find this project useful, please give it a **star** on GitHub! ⭐  

---

### 📌 License  
This project is licensed under the **MIT License**.  

---

🚀 **Let's build the future of AI-powered conversations!** 🤖✨  
```
