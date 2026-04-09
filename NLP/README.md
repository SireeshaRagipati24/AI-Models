# 📩 SMS Spam Classifier — NLP with Naive Bayes

> **Teaching a machine to read between the lines** — using Natural Language Processing to automatically detect spam SMS messages with ~98% accuracy.

---

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Domain** | Natural Language Processing (NLP) |
| **Algorithm** | Multinomial Naive Bayes |
| **Dataset** | SMS Spam Collection — 5,572 real SMS messages |
| **Task** | Binary Text Classification (Ham vs Spam) |
| **Vectorization** | Bag of Words (CountVectorizer) |
| **Text Processing** | Regex · Lowercasing · Stopword Removal · Stemming |
| **Test Accuracy** | ~98% |

---

## 🎯 Business Objective

> Build an intelligent SMS filter that **automatically detects spam messages** — protecting users from fraud, phishing, and unwanted promotions in real time.

**Real-world applications:**
- 📱 Mobile carrier spam filters
- 📧 Email spam detection (Gmail, Outlook)
- 🏦 Bank fraud SMS alert systems
- 💬 WhatsApp / Telegram spam detection

---

## 🧠 Why Naive Bayes for Text Classification?

| Algorithm | Why Considered | Limitation |
|-----------|---------------|------------|
| **Multinomial Naive Bayes** ⭐ | Fast, works perfectly with word counts, handles high-dimensional sparse data | Assumes word independence |
| Logistic Regression | Good baseline | Slower on large vocabularies |
| SVM | High accuracy | Computationally heavy on text |
| Deep Learning (BERT) | State-of-the-art | Needs much more data & compute |

> 🔑 **Multinomial Naive Bayes** is the industry go-to for text classification — it works natively with word count features and delivers near state-of-the-art results with minimal compute.

---

## 🗺️ Project Workflow

```
Load Raw SMS Data (5,572 messages)
           ↓
Exploratory Data Analysis
  → Class distribution (Ham vs Spam)
  → Message length analysis
  → Top spam vocabulary words
           ↓
Text Preprocessing Pipeline
  → Remove special characters (regex)
  → Convert to lowercase
  → Tokenize (split into words)
  → Remove stopwords (NLTK)
  → Stem to root form (PorterStemmer)
           ↓
Vectorization → CountVectorizer (Bag of Words)
  → Document-Term Matrix: (5572 messages × 5000 words)
           ↓
Label Encoding → ham=0, spam=1
           ↓
Train/Test Split → 70% train | 30% test
           ↓
Train Multinomial Naive Bayes (alpha=1.0 Laplace smoothing)
           ↓
Evaluate → Accuracy · Precision · Recall · F1 · Confusion Matrix
           ↓
Save Model + Vectorizer (joblib)
           ↓
Real-World Prediction on New Messages
```

---

## 🔧 NLP Preprocessing Pipeline

Raw text cannot be fed directly to ML models. Each message goes through:

| Step | Input Example | Output Example | Purpose |
|------|--------------|----------------|---------|
| Remove special chars | `'Win £1000!!!'` | `'Win      '` | Numbers/symbols = noise |
| Lowercase | `'FREE PRIZE'` | `'free prize'` | Case consistency |
| Tokenize | `'free prize'` | `['free','prize']` | Split into words |
| Remove stopwords | `['you','have','won']` | `['won']` | Remove meaningless words |
| Stemming | `['winning','winner']` | `['win','win']` | Reduce to root form |
| Rejoin | `['win','prize']` | `'win prize'` | Ready for vectorizer |

---

## 💡 Bag of Words — How It Works

```
Vocabulary (unique words across all messages):
  ['call', 'claim', 'free', 'prize', 'win', ...]

Message: "free prize win"
Vector:  [  0,    1,    1,    1,    1, ...]
          ↑                              
         'call' not present  
```

Each message becomes a **sparse numeric vector** of word counts — called a **Document-Term Matrix**.

---

## 📊 Key Results

| Metric | Ham (Legit) | Spam | Overall |
|--------|------------|------|---------|
| Precision | ~99% | ~94% | — |
| Recall | ~99% | ~92% | — |
| F1 Score | ~99% | ~93% | — |
| **Accuracy** | — | — | **~98%** |

**Confusion Matrix:**
```
                Predicted Ham    Predicted Spam
Actual Ham   |   True Neg (TN)  |  False Pos (FP)  ← Legit msg deleted (bad!)
Actual Spam  |   False Neg (FN) |  True Pos  (TP)  ← Spam caught ✅
```

> The model catches **92%+ of all spam** while incorrectly flagging very few legitimate messages.

---

## 📂 Project Structure

```
NLP_Spam_Classifier/
│
├── 📓 NLP_spam_classifier.ipynb       # Full NLP notebook (end-to-end)
├── 📊 SMSSpamCollection               # Raw dataset (tab-separated, 5572 msgs)
├── 🤖 spam_classifier_model.pkl       # Saved Naive Bayes model
├── 🔤 count_vectorizer.pkl            # Saved CountVectorizer (same vocab as training)
└── 📝 README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **NLTK** | Stopwords list + PorterStemmer |
| **Scikit-learn** | CountVectorizer · MultinomialNB · Metrics |
| **Pandas / NumPy** | Data loading and manipulation |
| **Matplotlib / Seaborn** | EDA visualisations |
| **Joblib** | Model + vectorizer serialization |
| **Regex (`re`)** | Text cleaning |

---

## ▶️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/SireeshaRagipati24/Machine-Learning.git
cd Machine-Learning
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib
```

### 3. Download NLTK data
```python
import nltk
nltk.download('stopwords')
```

### 4. Run the notebook
```bash
jupyter notebook NLP_spam_classifier.ipynb
```

---

## 🔮 Predict on New Messages

```python
from joblib import load
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load saved model and vectorizer
model = load('spam_classifier_model.pkl')
cv    = load('count_vectorizer.pkl')

def predict_spam(message):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Preprocess
    text = re.sub('[^a-zA-Z]', ' ', message).lower().split()
    text = ' '.join([ps.stem(w) for w in text if w not in stop_words])
    
    # Vectorize + Predict
    vector = cv.transform([text]).toarray()
    result = model.predict(vector)[0]
    return '🚨 SPAM' if result == 1 else '✅ HAM'

# Test it!
print(predict_spam("Congratulations! You've won a £1000 prize. Call now!"))
# → 🚨 SPAM

print(predict_spam("Hey, are you free for lunch tomorrow?"))
# → ✅ HAM
```

---

## 💡 Key NLP Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Text Cleaning | Regex — remove non-alphabetic chars |
| Normalisation | Lowercasing for consistency |
| Tokenization | `.split()` on cleaned text |
| Stopword Removal | NLTK English stopwords list |
| Stemming | PorterStemmer → root form |
| Bag of Words | CountVectorizer → sparse matrix |
| Naive Bayes | Probabilistic classifier (Bayes' Theorem) |
| Laplace Smoothing | `alpha=1.0` — handles unseen words |
| Model Persistence | joblib for both model + vectorizer |

---

## 🚀 Future Improvements

- [ ] Try **TF-IDF Vectorizer** — penalises overly common words
- [ ] Use **n-grams (bigrams/trigrams)** to capture `'free prize'`, `'click now'`
- [ ] Compare with **Logistic Regression** and **SVM**
- [ ] Handle class imbalance with **SMOTE** oversampling
- [ ] Build a **Streamlit web app** for interactive spam checking
- [ ] Fine-tune **BERT / DistilBERT** for higher accuracy

---

## 🙋‍♀️ About Me

**Sireesha Ragipati** — Aspiring Data Analyst passionate about turning raw data into intelligent decisions.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sireesha-ragipati-269a10244/)

---

*⭐ If you found this helpful, give it a star!*
