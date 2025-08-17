# 📧 Spam-Ham Classifier

A machine learning project that classifies text messages as **Spam** or **Ham** using:
- 🔍 TF-IDF vectorization
- 🏷️ Keyword flags for spammy terms
- 📊 Statistical text features
- 🤖 Logistic Regression classifier

It also provides:
- 📈 Visualizations of spam vs ham distribution
- 🔤 Keyword occurrence plots
- ☁️ Word clouds
- 🔥 Confusion matrix heatmap

---

## 🚀 Features

### 🧹 Data Preprocessing
- Removes URLs, emails, punctuation
- Lowercases and lemmatizes text
- Filters stopwords

### 🛠️ Feature Engineering
- TF-IDF (unigrams + bigrams)
- Character count, word count
- Average word length
- Number of exclamation marks, digits, uppercase letters

### 🧠 Model Training
- Logistic Regression classifier
- Auto-discovers top spam keywords
- Saves model, TF-IDF vectorizer, and keyword list

### 🧪 Prediction
- CLI tool to classify new messages as Spam or Ham

### 📊 Visualizations
- Spam vs Ham bar chart
- Word count distribution
- Keyword frequency bar chart
- Spam word cloud
- Confusion matrix heatmap

---

## 📦 Requirements

Install the following Python packages before running the project:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn joblib wordcloud
