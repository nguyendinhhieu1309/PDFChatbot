
# 📄 PDF Chatbot for Document Analysis

**Developed by [Nguyen Dinh Hieu]**  
[hieundhe180318@fpt.edu.vn]

---

## ✨ Abstract

This project introduces a PDF chatbot application that processes PDF documents, enabling users to interact with the content through natural language queries. The chatbot remembers the context of conversations and the history of interactions, allowing users to ask follow-up questions and receive relevant answers. It uses machine learning models from Google Generative AI and embeddings to create a knowledge base from the documents and provides insightful responses to the user's queries.

---

## 📚 Table of Contents

- [📄 Project Overview](#project-overview)
- [🛠️ Environment Setup](#environment-setup)
- [💻 Installation](#installation)
- [🚀 Features](#features)
- [🔍 Testing](#testing)
- [🤝 Acknowledgements](#acknowledgements)

---

## 📄 Project Overview

The PDF Chatbot application is designed to load, process, and interact with content from PDF documents. By using Google Generative AI models and embeddings, this application provides responses to user queries and stores conversation history, which enables a more personalized interaction. The chatbot is built with Flask and uses a Chroma vector store for managing document embeddings and conversation history.

---

## 🛠️ Environment Setup

This project uses the following libraries:

| **Library** | **Version** |
|:------------|:-----------|
| [Flask](https://flask.palletsprojects.com/) | 2.1.2 |
| [Langchain](https://www.langchain.com) | 0.0.121 |
| [Google Generative AI](https://cloud.google.com/blog/topics/ai-machine-learning) | 1.0.0 |
| [Chroma](https://www.trychroma.com/) | 0.3.0 |
| [PyPDF2](https://pythonhosted.org/PyPDF2/) | 1.26.0 |

---

## 💻 Installation

1. **Create a Virtual Environment**  
   ```bash
   python -m venv chatbot_env
   source chatbot_env/bin/activate  # On Windows use `chatbot_env\Scripts\activate`
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Clone the Repository**  
   ```bash
   git clone https://github.com/nguyendinhhieu1309/PDFChatbot.git
   cd pdf-chatbot
   ```

---

## 🚀 Features

### 1. **PDF Document Processing**
   - Loads multiple PDF files from a specified directory.
   - Splits large documents into smaller chunks for efficient processing.

### 2. **Contextual Chatbot**
   - The chatbot remembers the conversation context and history of previous interactions.
   - The application uses embeddings and vector stores to allow the bot to provide relevant and insightful answers based on the document's content.

### 3. **Customizable System Prompt**
   - You can define a custom system prompt for the chatbot to follow, tailored to your specific needs.

### 4. **Query Answering**
   - The chatbot answers user queries by analyzing the PDF content.
   - It processes follow-up questions by maintaining the context of prior conversations.

### 5. **Conversation History**
   - The chatbot remembers and stores the entire conversation history.
   - Chat history is stored in a vector store, ensuring that the bot can reference past interactions to improve future responses.

---

## 🔍 Testing

1. **Run the Application**  
   Start the Flask server by running the following command:
   ```bash
   python chatbot.py
   ```

2. **Interact with the Chatbot**  
   Open your browser and go to `http://127.0.0.1:5000/` to begin interacting with the chatbot. You can ask questions about the PDFs you have loaded, and the chatbot will provide answers based on the content of those PDFs.

---

## 🤝 Acknowledgements

This project utilizes Langchain, Google Generative AI, and Chroma for managing documents, embeddings, and conversation history. Special thanks to the open-source community for their contributions to these libraries.

---

### 🎨 Demo

Here is a screenshot of the chatbot in action:
![image](https://github.com/user-attachments/assets/7f3670e0-b877-48e7-afe2-adbcdc14d21f)

