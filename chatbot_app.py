"""
🤖 Python | NLP | ML Basics Chatbot
A Streamlit chatbot that ONLY answers questions about Python, NLP, and ML Basics.
All other questions get a polite "Sorry" response.

To run: streamlit run chatbot_app.py
Install:  pip install streamlit scikit-learn nltk
"""

import streamlit as st
import re
import random
from datetime import datetime

# ─────────────────────────────────────────────
#  KNOWLEDGE BASE  – Python, NLP, ML Basics
# ─────────────────────────────────────────────

KNOWLEDGE_BASE = {

    # ── PYTHON ──────────────────────────────
    "what is python": (
        "**Python** is a high-level, interpreted, general-purpose programming language created by "
        "Guido van Rossum in 1991. It emphasises code readability, uses indentation instead of "
        "braces, and supports multiple paradigms (procedural, OOP, functional). "
        "It is the #1 language for Data Science, ML, and NLP because of its rich ecosystem "
        "(NumPy, Pandas, scikit-learn, NLTK, spaCy, TensorFlow, PyTorch)."
    ),
    "python data types": (
        "Python's built-in data types include:\n"
        "- **int** – integers (e.g. `42`)\n"
        "- **float** – decimals (e.g. `3.14`)\n"
        "- **str** – strings (e.g. `'hello'`)\n"
        "- **bool** – `True` / `False`\n"
        "- **list** – mutable ordered sequence `[1,2,3]`\n"
        "- **tuple** – immutable ordered sequence `(1,2,3)`\n"
        "- **dict** – key-value pairs `{'key':'value'}`\n"
        "- **set** – unique unordered elements `{1,2,3}`\n"
        "- **NoneType** – represents `None`"
    ),
    "python list": (
        "A **list** is a mutable, ordered collection in Python.\n"
        "```python\nfruits = ['apple', 'banana', 'cherry']\n"
        "fruits.append('date')   # add item\n"
        "fruits.remove('apple')  # remove item\n"
        "print(fruits[0])        # indexing\n"
        "print(fruits[1:3])      # slicing\n```"
    ),
    "python dictionary": (
        "A **dict** stores key-value pairs and is mutable.\n"
        "```python\nstudent = {'name': 'Alice', 'age': 22, 'grade': 'A'}\n"
        "print(student['name'])       # 'Alice'\n"
        "student['age'] = 23          # update\n"
        "student['email'] = 'a@b.com' # add new key\n"
        "del student['grade']         # delete\n```"
    ),
    "python loop": (
        "Python has two main loops:\n\n"
        "**for loop** – iterates over a sequence:\n"
        "```python\nfor i in range(5):\n    print(i)  # 0 1 2 3 4\n```\n\n"
        "**while loop** – runs while condition is True:\n"
        "```python\nx = 0\nwhile x < 5:\n    print(x)\n    x += 1\n```"
    ),
    "python function": (
        "A **function** is a reusable block of code defined with `def`:\n"
        "```python\ndef greet(name, greeting='Hello'):\n"
        "    return f'{greeting}, {name}!'\n\n"
        "print(greet('Alice'))          # Hello, Alice!\n"
        "print(greet('Bob', 'Hi'))      # Hi, Bob!\n```\n"
        "Python also supports **lambda** (anonymous) functions:\n"
        "```python\nsquare = lambda x: x**2\nprint(square(5))  # 25\n```"
    ),
    "python class": (
        "**OOP in Python** – classes are blueprints for objects:\n"
        "```python\nclass Animal:\n"
        "    def __init__(self, name, sound):\n"
        "        self.name = name\n"
        "        self.sound = sound\n\n"
        "    def speak(self):\n"
        "        return f'{self.name} says {self.sound}!'\n\n"
        "dog = Animal('Rex', 'Woof')\n"
        "print(dog.speak())  # Rex says Woof!\n```"
    ),
    "python exception": (
        "Handle errors with **try / except / finally**:\n"
        "```python\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError as e:\n"
        "    print(f'Error: {e}')\nexcept (TypeError, ValueError):\n"
        "    print('Type or value error')\nfinally:\n    print('Always runs')\n```\n"
        "Raise your own exception: `raise ValueError('Bad input')`"
    ),
    "python list comprehension": (
        "**List comprehensions** are concise ways to create lists:\n"
        "```python\n# Basic\nsquares = [x**2 for x in range(10)]\n\n"
        "# With condition\nevens = [x for x in range(20) if x % 2 == 0]\n\n"
        "# Nested\nmatrix = [[i*j for j in range(3)] for i in range(3)]\n```"
    ),
    "python numpy": (
        "**NumPy** is the fundamental package for numerical computing:\n"
        "```python\nimport numpy as np\n\narr = np.array([1, 2, 3, 4, 5])\nprint(arr.mean())   # 3.0\nprint(arr.std())    # 1.41\n\n"
        "# 2D array\nmatrix = np.array([[1,2],[3,4]])\nprint(matrix.shape)  # (2, 2)\nprint(matrix.T)      # transpose\n```"
    ),
    "python pandas": (
        "**Pandas** is the go-to library for data manipulation:\n"
        "```python\nimport pandas as pd\n\n"
        "df = pd.read_csv('data.csv')\nprint(df.head())          # first 5 rows\nprint(df.describe())      # stats\n"
        "print(df['col'].value_counts())\n\ndf.dropna(inplace=True)   # remove nulls\ndf['new'] = df['a'] + df['b']\n```"
    ),

    # ── NLP ──────────────────────────────────
    "what is nlp": (
        "**Natural Language Processing (NLP)** is a subfield of AI that enables computers to "
        "understand, interpret, generate, and respond to human language. Key tasks include:\n"
        "- Tokenization, POS Tagging, NER\n"
        "- Sentiment Analysis\n"
        "- Machine Translation\n"
        "- Text Summarisation\n"
        "- Question Answering\n"
        "- Speech Recognition\n\n"
        "Popular Python libraries: **NLTK**, **spaCy**, **Transformers (HuggingFace)**"
    ),
    "tokenization": (
        "**Tokenization** splits text into smaller units (tokens) — words or subwords.\n"
        "```python\nimport nltk\nnltk.download('punkt')\nfrom nltk.tokenize import word_tokenize, sent_tokenize\n\n"
        "text = 'NLP is amazing. I love learning it!'\nprint(word_tokenize(text))\n# ['NLP', 'is', 'amazing', '.', 'I', 'love', 'learning', 'it', '!']\n"
        "print(sent_tokenize(text))\n# ['NLP is amazing.', 'I love learning it!']\n```"
    ),
    "stop words": (
        "**Stop words** are common words (is, the, a, in…) that carry little meaning and are "
        "often removed before NLP tasks:\n"
        "```python\nfrom nltk.corpus import stopwords\nfrom nltk.tokenize import word_tokenize\n\n"
        "stop = set(stopwords.words('english'))\ntokens = word_tokenize('This is a great NLP example')\n"
        "filtered = [w for w in tokens if w.lower() not in stop]\nprint(filtered)  # ['great', 'NLP', 'example']\n```"
    ),
    "stemming": (
        "**Stemming** reduces words to their base/root form by chopping suffixes:\n"
        "```python\nfrom nltk.stem import PorterStemmer\nps = PorterStemmer()\n\n"
        "words = ['running', 'runner', 'runs', 'easily', 'fairly']\nfor w in words:\n    print(f'{w} -> {ps.stem(w)}')\n"
        "# running -> run, runner -> runner, easily -> easili\n```\n"
        "**Note:** Stemming may produce non-real words. Use **Lemmatization** for real words."
    ),
    "lemmatization": (
        "**Lemmatization** reduces words to their dictionary form (lemma):\n"
        "```python\nfrom nltk.stem import WordNetLemmatizer\nnltk.download('wordnet')\n\n"
        "lemmatizer = WordNetLemmatizer()\nprint(lemmatizer.lemmatize('running', pos='v'))  # run\n"
        "print(lemmatizer.lemmatize('better', pos='a'))   # good\n"
        "print(lemmatizer.lemmatize('cats'))               # cat\n```\n"
        "Unlike stemming, lemmatization always produces valid words."
    ),
    "pos tagging": (
        "**POS (Part-of-Speech) Tagging** labels each word with its grammatical role:\n"
        "```python\nimport nltk\nnltk.download('averaged_perceptron_tagger')\n\n"
        "tokens = word_tokenize('The quick brown fox jumps')\ntags = nltk.pos_tag(tokens)\nprint(tags)\n"
        "# [('The','DT'),('quick','JJ'),('brown','JJ'),('fox','NN'),('jumps','VBZ')]\n```\n"
        "Common tags: **NN**=Noun, **VB**=Verb, **JJ**=Adjective, **RB**=Adverb, **DT**=Determiner"
    ),
    "named entity recognition": (
        "**NER** identifies named entities (people, places, organisations) in text:\n"
        "```python\nimport spacy\nnlp = spacy.load('en_core_web_sm')\n\n"
        "doc = nlp('Apple was founded by Steve Jobs in Cupertino, California.')\nfor ent in doc.ents:\n"
        "    print(ent.text, ent.label_)\n"
        "# Apple ORG | Steve Jobs PERSON | Cupertino GPE | California GPE\n```"
    ),
    "bag of words": (
        "**Bag of Words (BoW)** represents text as word frequency counts, ignoring order:\n"
        "```python\nfrom sklearn.feature_extraction.text import CountVectorizer\n\n"
        "corpus = ['I love NLP', 'NLP is fun', 'I love coding']\nvec = CountVectorizer()\nX = vec.fit_transform(corpus)\nprint(vec.get_feature_names_out())\n"
        "print(X.toarray())\n```\n"
        "Simple but effective for many classification tasks. Limitation: ignores word order and semantics."
    ),
    "tfidf": (
        "**TF-IDF** (Term Frequency-Inverse Document Frequency) scores words by importance:\n"
        "- **TF** = word frequency in a document\n"
        "- **IDF** = log(total docs / docs containing word) — penalises common words\n"
        "```python\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n\n"
        "corpus = ['I love NLP', 'NLP is great', 'Machine learning is fun']\nvec = TfidfVectorizer()\nX = vec.fit_transform(corpus)\nprint(X.toarray())\n```"
    ),
    "word embeddings": (
        "**Word Embeddings** map words to dense numeric vectors preserving semantic meaning:\n"
        "- Similar words are close in vector space\n"
        "- Famous example: king − man + woman ≈ queen\n\n"
        "Popular models: **Word2Vec**, **GloVe**, **FastText**\n"
        "```python\nfrom gensim.models import Word2Vec\n\nsentences = [['I','love','NLP'],['deep','learning','is','fun']]\nmodel = Word2Vec(sentences, vector_size=100, window=5, min_count=1)\n"
        "print(model.wv['NLP'])        # 100-dim vector\nprint(model.wv.most_similar('NLP'))\n```"
    ),
    "sentiment analysis": (
        "**Sentiment Analysis** classifies text as positive, negative, or neutral:\n"
        "```python\nfrom nltk.sentiment import SentimentIntensityAnalyzer\nnltk.download('vader_lexicon')\n\n"
        "sia = SentimentIntensityAnalyzer()\nsentences = ['I love this product!', 'This is terrible.', 'It is okay.']\n"
        "for s in sentences:\n    score = sia.polarity_scores(s)\n    print(s, '->', score['compound'])\n"
        "# > 0.05 = positive, < -0.05 = negative\n```"
    ),
    "text preprocessing": (
        "A typical **NLP preprocessing pipeline**:\n"
        "1. **Lowercasing** – `text.lower()`\n"
        "2. **Remove punctuation/special chars** – regex\n"
        "3. **Tokenization** – split into words\n"
        "4. **Stop word removal** – filter common words\n"
        "5. **Stemming or Lemmatization** – normalise words\n"
        "6. **Vectorisation** – BoW / TF-IDF / Embeddings\n\n"
        "```python\nimport re\nfrom nltk.tokenize import word_tokenize\nfrom nltk.corpus import stopwords\nfrom nltk.stem import WordNetLemmatizer\n\n"
        "def preprocess(text):\n    text = text.lower()\n    text = re.sub(r'[^a-z\\s]', '', text)\n"
        "    tokens = word_tokenize(text)\n    stop = set(stopwords.words('english'))\n    tokens = [t for t in tokens if t not in stop]\n"
        "    lem = WordNetLemmatizer()\n    return [lem.lemmatize(t) for t in tokens]\n```"
    ),

    # ── ML BASICS ────────────────────────────
    "what is machine learning": (
        "**Machine Learning (ML)** is a subset of AI where systems learn patterns from data "
        "without being explicitly programmed. Categories:\n"
        "- **Supervised Learning** – learn from labelled data (classification, regression)\n"
        "- **Unsupervised Learning** – find patterns in unlabelled data (clustering, dimensionality reduction)\n"
        "- **Reinforcement Learning** – agent learns by trial and reward/penalty\n\n"
        "Key Python library: **scikit-learn** (`sklearn`)"
    ),
    "supervised learning": (
        "**Supervised Learning** trains on (input, label) pairs to predict labels for new inputs.\n\n"
        "- **Classification** – output is a category (spam/not spam, cat/dog)\n"
        "- **Regression** – output is a number (house price, temperature)\n\n"
        "```python\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nmodel = LogisticRegression()\nmodel.fit(X_train, y_train)\nprint(model.score(X_test, y_test))\n```"
    ),
    "unsupervised learning": (
        "**Unsupervised Learning** finds hidden patterns in data without labels.\n\n"
        "- **Clustering** – group similar data points (K-Means, DBSCAN)\n"
        "- **Dimensionality Reduction** – reduce features (PCA, t-SNE)\n"
        "- **Association Rules** – find item relationships (Apriori)\n\n"
        "```python\nfrom sklearn.cluster import KMeans\n\nkmeans = KMeans(n_clusters=3, random_state=42)\nkmeans.fit(X)\nlabels = kmeans.labels_\ncenters = kmeans.cluster_centers_\n```"
    ),
    "linear regression": (
        "**Linear Regression** models the relationship between features and a continuous target:\n"
        "**y = β₀ + β₁x₁ + β₂x₂ + … + ε**\n"
        "```python\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error, r2_score\n\n"
        "model = LinearRegression()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\n\n"
        "print('MSE:', mean_squared_error(y_test, y_pred))\nprint('R²:', r2_score(y_test, y_pred))\nprint('Coefficients:', model.coef_)\n```"
    ),
    "logistic regression": (
        "**Logistic Regression** is a classification algorithm using the sigmoid function:\n"
        "**P(y=1) = 1 / (1 + e^(-z))**\n"
        "```python\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score, classification_report\n\n"
        "model = LogisticRegression(max_iter=200)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\n\n"
        "print(accuracy_score(y_test, y_pred))\nprint(classification_report(y_test, y_pred))\n```\n"
        "Despite the name, it's a **classifier** not a regressor!"
    ),
    "decision tree": (
        "**Decision Tree** splits data based on feature thresholds to form a tree structure:\n"
        "```python\nfrom sklearn.tree import DecisionTreeClassifier, plot_tree\nimport matplotlib.pyplot as plt\n\n"
        "dt = DecisionTreeClassifier(max_depth=4, random_state=42)\ndt.fit(X_train, y_train)\nprint('Accuracy:', dt.score(X_test, y_test))\n\n"
        "# Visualise\nplt.figure(figsize=(15,8))\nplot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True)\nplt.show()\n```\n"
        "**Pros:** Interpretable. **Cons:** Prone to overfitting (use Random Forest instead)."
    ),
    "random forest": (
        "**Random Forest** = ensemble of decision trees trained on random data subsets (bagging):\n"
        "```python\nfrom sklearn.ensemble import RandomForestClassifier\n\n"
        "rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)\nrf.fit(X_train, y_train)\nprint('Accuracy:', rf.score(X_test, y_test))\n\n"
        "# Feature importance\nimportances = rf.feature_importances_\n```\n"
        "✅ More accurate than single trees, resistant to overfitting, handles missing values."
    ),
    "svm": (
        "**Support Vector Machine (SVM)** finds the optimal hyperplane that maximises the margin between classes:\n"
        "```python\nfrom sklearn.svm import SVC\nfrom sklearn.preprocessing import StandardScaler\n\n"
        "scaler = StandardScaler()\nX_train_s = scaler.fit_transform(X_train)\nX_test_s = scaler.transform(X_test)\n\n"
        "svm = SVC(kernel='rbf', C=1.0, gamma='scale')\nsvm.fit(X_train_s, y_train)\nprint('Accuracy:', svm.score(X_test_s, y_test))\n```\n"
        "Kernels: **linear**, **rbf** (radial), **poly**. Always scale features before SVM!"
    ),
    "k nearest neighbors": (
        "**K-Nearest Neighbors (KNN)** classifies a point by majority vote of its K nearest neighbours:\n"
        "```python\nfrom sklearn.neighbors import KNeighborsClassifier\n\n"
        "knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')\nknn.fit(X_train, y_train)\nprint('Accuracy:', knn.score(X_test, y_test))\n```\n"
        "**Choosing K:** small K = complex boundary (overfit), large K = smooth boundary (underfit). Use cross-validation!"
    ),
    "naive bayes": (
        "**Naive Bayes** uses Bayes' theorem with the \"naive\" assumption of feature independence:\n"
        "**P(class|features) ∝ P(class) × ∏ P(feature|class)**\n"
        "```python\nfrom sklearn.naive_bayes import MultinomialNB\n\n"
        "nb = MultinomialNB()\nnb.fit(X_train, y_train)\nprint('Accuracy:', nb.score(X_test, y_test))\n```\n"
        "Types: **GaussianNB** (continuous), **MultinomialNB** (counts/NLP), **BernoulliNB** (binary). Very fast and great for text classification!"
    ),
    "overfitting underfitting": (
        "**Overfitting** – model learns training data too well, fails on new data (high variance)\n"
        "**Underfitting** – model too simple, can't even fit training data (high bias)\n\n"
        "**Solutions for overfitting:**\n"
        "- More training data\n"
        "- Regularisation (L1 Lasso, L2 Ridge)\n"
        "- Dropout (neural nets)\n"
        "- Reduce model complexity\n"
        "- Cross-validation\n\n"
        "**Solutions for underfitting:**\n"
        "- Increase model complexity\n"
        "- Add more features\n"
        "- Train longer / remove regularisation"
    ),
    "train test split": (
        "**Train-Test Split** partitions data to evaluate model performance on unseen data:\n"
        "```python\nfrom sklearn.model_selection import train_test_split\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n    X, y,\n    test_size=0.2,    # 20% test\n    random_state=42,  # reproducibility\n    stratify=y        # preserve class ratios\n)\nprint(f'Train: {X_train.shape}, Test: {X_test.shape}')\n```\n"
        "Typical splits: **70/30** or **80/20**. Use **stratify** for imbalanced datasets."
    ),
    "cross validation": (
        "**Cross-Validation (CV)** gives a more reliable performance estimate than a single split:\n"
        "```python\nfrom sklearn.model_selection import cross_val_score, KFold\n\n"
        "model = RandomForestClassifier(n_estimators=100)\nkf = KFold(n_splits=5, shuffle=True, random_state=42)\n\n"
        "scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')\nprint(f'CV Scores: {scores}')\nprint(f'Mean: {scores.mean():.3f} ± {scores.std():.3f}')\n```\n"
        "**k-fold CV** is the most common. Use **StratifiedKFold** for classification."
    ),
    "confusion matrix": (
        "**Confusion Matrix** shows True/False Positives and Negatives:\n"
        "```\n          Predicted\n          Pos  Neg\nActual Pos [TP  FN]\n       Neg [FP  TN]\n```\n"
        "```python\nfrom sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\nimport matplotlib.pyplot as plt\n\n"
        "cm = confusion_matrix(y_test, y_pred)\ndisp = ConfusionMatrixDisplay(cm, display_labels=class_names)\ndisp.plot()\nplt.show()\n```\n"
        "From the matrix:\n- **Accuracy** = (TP+TN)/(TP+TN+FP+FN)\n- **Precision** = TP/(TP+FP)\n- **Recall** = TP/(TP+FN)\n- **F1** = 2×(Precision×Recall)/(Precision+Recall)"
    ),
    "feature scaling": (
        "**Feature Scaling** normalises numeric features for algorithms sensitive to scale (SVM, KNN, Neural Nets):\n\n"
        "**StandardScaler** – zero mean, unit variance (z-score):\n"
        "```python\nfrom sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)  # use fit from training!\n```\n\n"
        "**MinMaxScaler** – scales to [0,1]:\n"
        "```python\nfrom sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()\nX_scaled = scaler.fit_transform(X)\n```"
    ),
    "neural network": (
        "**Neural Networks** consist of layers of interconnected neurons:\n"
        "- **Input layer** – receives features\n"
        "- **Hidden layers** – learn representations\n"
        "- **Output layer** – produces predictions\n\n"
        "```python\nfrom sklearn.neural_network import MLPClassifier\n\n"
        "mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',\n"
        "                    max_iter=300, random_state=42)\nmlp.fit(X_train, y_train)\nprint('Accuracy:', mlp.score(X_test, y_test))\n```\n"
        "For deep learning use **TensorFlow/Keras** or **PyTorch**."
    ),
    "gradient descent": (
        "**Gradient Descent** iteratively updates model parameters to minimise a loss function:\n\n"
        "**θ = θ - α × ∇Loss(θ)**  (α = learning rate)\n\n"
        "Variants:\n"
        "- **Batch GD** – uses all data (slow, stable)\n"
        "- **Stochastic GD (SGD)** – one sample at a time (fast, noisy)\n"
        "- **Mini-batch GD** – subset of data (best of both)\n\n"
        "```python\nfrom sklearn.linear_model import SGDClassifier\n\n"
        "sgd = SGDClassifier(loss='hinge', learning_rate='optimal', max_iter=1000)\nsgd.fit(X_train, y_train)\n```"
    ),
    "kmeans clustering": (
        "**K-Means Clustering** partitions data into K clusters by minimising intra-cluster variance:\n\n"
        "Algorithm: 1) Init K centroids → 2) Assign points to nearest centroid → 3) Update centroids → repeat\n"
        "```python\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n\n"
        "# Find optimal K using elbow method\ninertias = []\nfor k in range(1, 11):\n    km = KMeans(n_clusters=k, random_state=42)\n    km.fit(X)\n    inertias.append(km.inertia_)\n\n"
        "# Fit final model\nkm = KMeans(n_clusters=3, random_state=42)\nlabels = km.fit_predict(X)\nprint('Silhouette Score:', silhouette_score(X, labels))\n```"
    ),
    "pca": (
        "**PCA (Principal Component Analysis)** reduces dimensionality by projecting data onto principal components:\n"
        "```python\nfrom sklearn.decomposition import PCA\nimport matplotlib.pyplot as plt\n\n"
        "pca = PCA(n_components=2)\nX_2d = pca.fit_transform(X_scaled)\n\n"
        "print('Explained variance:', pca.explained_variance_ratio_)\n\n"
        "plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='viridis')\nplt.xlabel('PC1'); plt.ylabel('PC2')\nplt.show()\n```\n"
        "PCA requires scaled features. Check `explained_variance_ratio_` to choose n_components."
    ),
    "regularization": (
        "**Regularisation** penalises model complexity to prevent overfitting:\n\n"
        "**L1 (Lasso)** – adds |β| penalty, drives some weights to zero (feature selection):\n"
        "```python\nfrom sklearn.linear_model import Lasso\nmodel = Lasso(alpha=0.1)\n```\n\n"
        "**L2 (Ridge)** – adds β² penalty, shrinks weights (never zero):\n"
        "```python\nfrom sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1.0)\n```\n\n"
        "**ElasticNet** – combines L1 and L2:\n"
        "```python\nfrom sklearn.linear_model import ElasticNet\nmodel = ElasticNet(alpha=0.1, l1_ratio=0.5)\n```"
    ),
    "hyperparameter tuning": (
        "**Hyperparameter Tuning** finds the best model configuration:\n\n"
        "**GridSearchCV** – exhaustive search:\n"
        "```python\nfrom sklearn.model_selection import GridSearchCV\n\n"
        "param_grid = {'n_estimators': [50,100,200], 'max_depth': [3,5,None]}\ngrid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')\ngrid.fit(X_train, y_train)\nprint('Best params:', grid.best_params_)\nprint('Best score:', grid.best_score_)\n```\n\n"
        "**RandomizedSearchCV** – random sample from distributions (faster)."
    ),

}

# ─────────────────────────────────────────────
#  INTENT MATCHING ENGINE
# ─────────────────────────────────────────────

INTENT_MAP = {
    "what is python": ["what is python","explain python","python language","about python","define python","python intro","tell me about python","python overview","python programming language"],
    "python data types": ["data type","int float str bool","variable type","python type","integer string","list tuple dict set"],
    "python list": ["python list","how to use list","list in python","append remove list","list methods"],
    "python dictionary": ["dictionary","dict","key value","python dict","hash map","hash table"],
    "python loop": ["loop","for loop","while loop","iteration","iterate","range loop","looping"],
    "python function": ["function","def ","lambda","return statement","python function","method","def keyword"],
    "python class": ["class","oop","object oriented","inheritance","__init__","self ","instance","object","constructor"],
    "python exception": ["exception","try except","error handling","raise","catch error","try block","finally"],
    "python list comprehension": ["list comprehension","comprehension","one liner list","[x for x"],
    "python numpy": ["numpy","np.array","ndarray","numerical python","matrix operation","array"],
    "python pandas": ["pandas","dataframe","pd.","read_csv","series","data manipulation"],
    "what is nlp": ["what is nlp","natural language processing","nlp overview","about nlp","define nlp","nlp introduction","explain nlp"],
    "tokenization": ["tokenize","tokenization","word_tokenize","sent_tokenize","token","split text","break text"],
    "stop words": ["stop word","stopword","remove common words","filter words","nltk stopwords"],
    "stemming": ["stemming","porter stemmer","stem word","word stem","truncate suffix","word root"],
    "lemmatization": ["lemmatization","lemmatize","wordnet lemma","base form","dictionary form"],
    "pos tagging": ["pos tag","part of speech","pos_tag","noun verb adjective tag","grammar tag","postag","part-of-speech"],
    "named entity recognition": ["ner","named entity","entity recognition","person place org","spacy entity","find entity"],
    "bag of words": ["bag of words","bow","countvectorizer","count vector","word frequency","term frequency"],
    "tfidf": ["tfidf","tf-idf","tf idf","term frequency inverse","tfidfvectorizer","idf"],
    "word embeddings": ["word embedding","word2vec","glove","fasttext","word vector","dense vector","semantic vector"],
    "sentiment analysis": ["sentiment","sentiment analysis","positive negative","emotion analysis","opinion mining","vader","polarity"],
    "text preprocessing": ["text preprocessing","preprocess text","nlp pipeline","clean text","text cleaning","preprocessing steps"],
    "what is machine learning": ["what is machine learning","what is ml","machine learning overview","about ml","define machine learning","ml introduction","explain ml","machine learning basics"],
    "supervised learning": ["supervised learning","supervised","labeled data","classification regression","supervised ml"],
    "unsupervised learning": ["unsupervised","clustering","no labels","unlabeled","unsupervised learning"],
    "linear regression": ["linear regression","regression","predict continuous","continuous output","lm","least squares"],
    "logistic regression": ["logistic regression","binary classification","sigmoid","log regression","logit"],
    "decision tree": ["decision tree","tree classifier","dt classifier","tree based","gini entropy","plot_tree"],
    "random forest": ["random forest","rf classifier","ensemble tree","bagging","feature importance"],
    "svm": ["svm","support vector","support vector machine","kernel","hyperplane","margin","svr","svc"],
    "k nearest neighbors": ["knn","k nearest","nearest neighbor","k-nn","kneighbors","k nearest neighbor"],
    "naive bayes": ["naive bayes","bayes","bayesian","gaussiannb","multinomial nb","text classifier naive"],
    "overfitting underfitting": ["overfitting","underfitting","overfit","underfit","variance bias","bias variance","high variance","high bias"],
    "train test split": ["train test split","split data","test size","training data","test data","split dataset"],
    "cross validation": ["cross validation","cv","k-fold","kfold","stratifiedkfold","cross_val_score"],
    "confusion matrix": ["confusion matrix","tp tn fp fn","precision recall","f1 score","classification report","accuracy metric","false positive"],
    "feature scaling": ["feature scaling","normalise","normalize","standardscaler","minmaxscaler","scale feature","standardize"],
    "neural network": ["neural network","mlp","multilayer perceptron","deep learning","hidden layer","activation function","neuron"],
    "gradient descent": ["gradient descent","sgd","optimiser","optimizer","learning rate","backpropagation","weight update"],
    "kmeans clustering": ["kmeans","k-means","k means","cluster","elbow method","silhouette","centroid"],
    "pca": ["pca","principal component","dimensionality reduction","reduce dimension","explained variance","projection"],
    "regularization": ["regularization","regularisation","lasso","ridge","elasticnet","l1 l2","penalty","overfitting regularise"],
    "hyperparameter tuning": ["hyperparameter","gridsearch","randomizedsearch","tuning","best params","cross validate model","model selection"],
}

GREETINGS = {
    "hi": "Hi there! 👋 I'm your **Python | NLP | ML Basics** assistant. Ask me anything about these three topics!",
    "hello": "Hello! 🤖 Ready to help with **Python**, **NLP**, or **ML Basics**. What would you like to learn?",
    "hey": "Hey! 👋 Ask me about **Python**, **NLP**, or **ML Basics** and I'll do my best to help!",
    "good morning": "Good morning! ☀️ Ready to learn some Python, NLP, or ML today?",
    "good afternoon": "Good afternoon! 🌤️ What Python, NLP, or ML topic can I help you with?",
    "good evening": "Good evening! 🌙 Let's dive into some Python, NLP, or ML concepts!",
    "thanks": "You're welcome! 😊 Feel free to ask more Python, NLP, or ML questions.",
    "thank you": "My pleasure! 🎉 Any more Python, NLP, or ML questions?",
    "help": (
        "Here's what I can help you with:\n\n"
        "🐍 **Python** – data types, lists, dicts, loops, functions, OOP, exceptions, NumPy, Pandas\n\n"
        "🔤 **NLP** – tokenization, stop words, stemming, lemmatization, POS tagging, NER, BoW, TF-IDF, Word Embeddings, Sentiment Analysis\n\n"
        "🤖 **ML Basics** – supervised/unsupervised, linear/logistic regression, decision trees, random forest, SVM, KNN, Naive Bayes, overfitting, train-test split, cross-validation, confusion matrix, scaling, neural networks, gradient descent, KMeans, PCA, regularization, hyperparameter tuning\n\n"
        "Just type your question!"
    ),
}

SORRY_RESPONSES = [
    "😔 Sorry, I can only answer questions about **Python**, **NLP**, and **ML Basics**. Please ask something related to these topics!",
    "🚫 That's outside my expertise! I'm specialised in **Python**, **NLP**, and **ML Basics** only.",
    "❌ I'm not able to help with that. My knowledge is limited to **Python programming**, **Natural Language Processing**, and **ML Basics**.",
    "⚠️ Sorry! I only know about **Python**, **NLP**, and **Machine Learning Basics**. Try asking about those topics!",
    "🤔 That topic isn't in my domain. I'm here to help with **Python**, **NLP**, and **ML Basics** exclusively.",
]


def match_intent(query: str):
    q = query.lower().strip()

    # Greetings
    for key in GREETINGS:
        if key in q:
            return GREETINGS[key]

    # Intent matching (score-based)
    best_key = None
    best_score = 0
    for intent_key, keywords in INTENT_MAP.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > best_score:
            best_score = score
            best_key = intent_key

    if best_score > 0 and best_key in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[best_key]

    return random.choice(SORRY_RESPONSES)


# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="PyNLP-ML Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

:root {
    --bg: #0d0f1a;
    --card: #151929;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --accent3: #22c55e;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e2d48;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg);
    color: var(--text);
}

/* Hide streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* Top header */
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a0a3d 50%, #0a2d2e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,229,255,0.06), transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(124,58,237,0.06), transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #00e5ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero p {
    color: var(--muted);
    margin: 0;
    font-size: 0.9rem;
}

/* Chat area */
.chat-area {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 4px;
}
.chat-area::-webkit-scrollbar { width: 4px; }
.chat-area::-webkit-scrollbar-track { background: var(--bg); }
.chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* Message bubbles */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}
.bubble-user {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    padding: 12px 18px;
    border-radius: 20px 20px 4px 20px;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3);
}
.bubble-bot {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 12px 18px;
    border-radius: 20px 20px 20px 4px;
    max-width: 85%;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar-bot {
    background: linear-gradient(135deg, #00e5ff33, #7c3aed33);
    border: 1px solid var(--accent);
    margin-right: 10px;
}
.avatar-user {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    margin-left: 10px;
}
.timestamp {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 4px;
    text-align: right;
}

/* Input */
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 14px 18px !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00e5ff, #0891b2) !important;
    color: #0d0f1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
.topic-chip {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    margin: 2px;
    cursor: default;
}
.stat-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    text-align: center;
}
.stat-num {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.stat-label { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

/* Code blocks */
code {
    background: #0d1b2a !important;
    color: #22c55e !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 4px;
    padding: 1px 5px;
}
pre {
    background: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "topic_counts" not in st.session_state:
    st.session_state.topic_counts = {"Python": 0, "NLP": 0, "ML": 0, "Other": 0}

def classify_topic(query):
    q = query.lower()
    python_kw = ["python","list","dict","loop","function","class","exception","numpy","pandas","lambda","comprehension"]
    nlp_kw = ["nlp","token","stop word","stem","lemma","pos tag","ner","named entity","bag of word","tfidf","embedding","sentiment","natural language"]
    ml_kw = ["machine learning","regression","classification","tree","forest","svm","knn","naive","cluster","pca","gradient","neural","overfit","cross valid","train test","confusion","scaling","regulariz","hyperparameter"]
    if any(k in q for k in python_kw): return "Python"
    if any(k in q for k in nlp_kw): return "NLP"
    if any(k in q for k in ml_kw): return "ML"
    return "Other"

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 PyNLP-ML Bot")
    st.markdown("A focused chatbot that **only** answers questions about:")
    st.markdown('<span class="topic-chip">🐍 Python</span> <span class="topic-chip">🔤 NLP</span> <span class="topic-chip">🤖 ML Basics</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.total_queries}</div><div class="stat-label">Total Queries</div></div>', unsafe_allow_html=True)
    with col2:
        answered = st.session_state.topic_counts["Python"] + st.session_state.topic_counts["NLP"] + st.session_state.topic_counts["ML"]
        st.markdown(f'<div class="stat-box"><div class="stat-num">{answered}</div><div class="stat-label">Answered</div></div>', unsafe_allow_html=True)

    st.markdown(f"🐍 Python: **{st.session_state.topic_counts['Python']}**")
    st.markdown(f"🔤 NLP: **{st.session_state.topic_counts['NLP']}**")
    st.markdown(f"🤖 ML: **{st.session_state.topic_counts['ML']}**")

    st.divider()
    st.markdown("#### 💡 Try Asking")
    sample_qs = [
        "What is Python?", "Explain tokenization", "What is linear regression?",
        "How does SVM work?", "What is TF-IDF?", "Explain overfitting",
        "What is NER?", "How does Random Forest work?", "Explain PCA",
        "What is gradient descent?"
    ]
    for q in sample_qs[:8]:
        st.markdown(f"• {q}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.topic_counts = {"Python": 0, "NLP": 0, "ML": 0, "Other": 0}
        st.rerun()

# ── Main area ────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🤖 Python | NLP | ML Basics Chatbot</h1>
    <p>Ask me anything about Python programming, Natural Language Processing, or Machine Learning basics. All other topics will be declined.</p>
</div>
""", unsafe_allow_html=True)

# Chat display
chat_html = '<div class="chat-area" id="chat-bottom">'
if not st.session_state.messages:
    chat_html += """
    <div style="text-align:center; padding: 60px 20px; color: #475569;">
        <div style="font-size:3rem; margin-bottom:12px">🤖</div>
        <div style="font-size:1.1rem; font-weight:600; color:#94a3b8">Start the conversation!</div>
        <div style="font-size:0.85rem; margin-top:6px">Ask a question about Python, NLP, or ML Basics</div>
    </div>
    """
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f"""
            <div class="msg-user">
                <div>
                    <div class="bubble-user">{msg['content']}</div>
                    <div class="timestamp">{msg['time']}</div>
                </div>
                <div class="avatar avatar-user">👤</div>
            </div>"""
        else:
            chat_html += f"""
            <div class="msg-bot">
                <div class="avatar avatar-bot">🤖</div>
                <div>
                    <div class="bubble-bot">{msg['content']}</div>
                    <div class="timestamp">{msg['time']}</div>
                </div>
            </div>"""
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# ── Input ────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Ask about Python, NLP, or ML Basics...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

if submitted and user_input.strip():
    query = user_input.strip()
    now = datetime.now().strftime("%H:%M")

    # Get response
    response = match_intent(query)

    # Convert markdown bold to HTML for chat bubbles
    def md_to_html(text):
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'```python\n(.*?)```', lambda m: f'<pre><code>{m.group(1)}</code></pre>', text, flags=re.DOTALL)
        text = re.sub(r'```\n(.*?)```', lambda m: f'<pre><code>{m.group(1)}</code></pre>', text, flags=re.DOTALL)
        text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
        text = text.replace('\n- ', '<br>• ').replace('\n', '<br>')
        return text

    # Save messages
    st.session_state.messages.append({"role": "user", "content": query, "time": now})
    st.session_state.messages.append({"role": "bot", "content": md_to_html(response), "time": now})

    # Update stats
    st.session_state.total_queries += 1
    topic = classify_topic(query)
    st.session_state.topic_counts[topic] += 1

    st.rerun()
