# 🤖 Python | NLP | ML Basics Chatbot

A **Streamlit-based intelligent chatbot** that answers questions related only to **Python Programming, Natural Language Processing (NLP), and Machine Learning Basics**.

If a user asks anything outside these domains, the chatbot politely declines and asks the user to stay within supported topics.

This project demonstrates **rule-based NLP intent detection, keyword matching, and a structured knowledge base** built using Python.

---

# 🚀 Features

* 💬 Interactive **chat interface**
* 🧠 **Intent matching engine** using keyword scoring
* 📚 Built-in **knowledge base** for:

  * Python programming
  * Natural Language Processing
  * Machine Learning fundamentals
* ⚡ Fast responses without external APIs
* 📊 **Session statistics dashboard**
* 🎨 Modern **custom Streamlit UI**
* 🔒 Restricted domain chatbot (answers only specific topics)

---

# 🧠 Supported Topics

## 🐍 Python

* Python Introduction
* Data Types
* Lists
* Dictionaries
* Loops
* Functions
* Classes (OOP)
* Exception Handling
* List Comprehensions
* NumPy
* Pandas

## 🔤 NLP (Natural Language Processing)

* Tokenization
* Stop Words
* Stemming
* Lemmatization
* POS Tagging
* Named Entity Recognition
* Bag of Words
* TF-IDF
* Word Embeddings
* Sentiment Analysis
* Text Preprocessing

## 🤖 Machine Learning Basics

* Machine Learning Overview
* Supervised Learning
* Unsupervised Learning
* Linear Regression
* Logistic Regression
* Decision Trees
* Random Forest
* Support Vector Machines (SVM)
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Overfitting & Underfitting
* Train/Test Split
* Cross Validation
* Confusion Matrix
* Feature Scaling
* Neural Networks
* Gradient Descent
* K-Means Clustering
* PCA (Principal Component Analysis)
* Regularization
* Hyperparameter Tuning

---

# 🏗 Project Architecture

```
User Query
    │
    ▼
Intent Matching Engine
    │
    ├── Greeting detection
    │
    ├── Keyword scoring against INTENT_MAP
    │
    ▼
Matched Intent
    │
    ▼
Knowledge Base
    │
    ▼
Formatted Response
    │
    ▼
Streamlit Chat UI
```

---

# 🛠 Tech Stack

| Technology   | Purpose                       |
| ------------ | ----------------------------- |
| Python       | Core programming language     |
| Streamlit    | Web interface                 |
| Regex        | Query preprocessing           |
| Scikit-learn | ML examples in knowledge base |
| NLTK         | NLP examples                  |
| HTML/CSS     | Custom UI styling             |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/pynlp-ml-chatbot.git
cd pynlp-ml-chatbot
```

Install dependencies:

```bash
pip install streamlit scikit-learn nltk
```

---

# ▶️ Run the Application

```bash
streamlit run chatbot_app.py
```

The app will open in your browser:

```
http://localhost:8501
```

---

# 💬 Example Queries

Try asking:

```
What is Python?
Explain tokenization
What is TF-IDF?
How does SVM work?
Explain overfitting
What is PCA?
What is gradient descent?
```

If the question is unrelated, the bot will respond politely with a restriction message.

---

# 📊 Chatbot Capabilities

| Capability          | Implementation          |
| ------------------- | ----------------------- |
| Intent Detection    | Keyword scoring         |
| Knowledge Retrieval | Rule-based mapping      |
| Chat Interface      | Streamlit UI            |
| Session Analytics   | Streamlit Session State |
| Domain Restriction  | Topic classifier        |

---

# 📁 Project Structure

```
pynlp-ml-chatbot
│
├── chatbot_app.py
├── README.md
└── requirements.txt
```

---

# 🎯 Learning Outcomes

This project demonstrates:

* Building a **rule-based NLP chatbot**
* Designing a **knowledge-based AI assistant**
* Implementing **intent matching logic**
* Creating **interactive Streamlit applications**
* Applying **basic NLP and ML concepts**

---

# 🔮 Future Improvements

* Add **transformer models (BERT / GPT)** for smarter responses
* Integrate **vector search (FAISS)**
* Add **voice interaction**
* Expand knowledge base
* Deploy using **Streamlit Cloud / Docker**

---



# ⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork it
🧠 Build your own AI chatbot
