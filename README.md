## Pipeline Scripts

### 01_data.py
Fetches (<100) recent English-language news articles from NewsAPI using a humanitarian-focused query.  
Extracts key fields (source, author, title, description, URL, publication date, content) and saves results to `input_data.csv`.  
Also performs initial sentiment analysis and rule-based categorization for downstream training.

---

### 02_rules_based.py
Loads preprocessed articles from `input_data.csv`.  
Detects relevant MSF countries using spaCy NER, filters the dataset accordingly, and outputs structured results.  
Also reports category and sentiment distributions.

---

### 03_nb_tfidf_cv.py
Loads the processed dataset.  
Trains a Naive Bayes classifier using TF-IDF features on existing category labels.  
Evaluates performance with cross-validation when possible and adds predicted categories to the output.

---

### 04_nb_w2v.py
Loads processed articles.  
Converts text into spaCy vector embeddings and trains a Logistic Regression classifier.  
Includes robust fallback handling for small datasets and outputs predicted categories alongside original data.

---

### 05_transformers.py
Loads processed articles.  
Fine-tunes a transformer model (DistilBERT) on existing category labels with safeguards for small or imbalanced datasets.  
Evaluates performance when possible and generates predicted categories for all articles.

> Note: Requires installation of transformers with PyTorch:
```bash
pip install transformers[torch]
