import joblib
import pandas as pd
from scr.preprocess import load_and_clean
from scr.features import add_keyword_flags, extract_basic_features, tfidf_vectorize
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# 1. Load & clean dataset
df = load_and_clean("data/raw/SMSSpamCollection")

# 2. Add features
df = add_keyword_flags(df)
df = extract_basic_features(df)

# 3. TF-IDF vectorization
X_tfidf, tfidf_vec = tfidf_vectorize(df['cleaned'])

# 4. Combine all numeric features with TF-IDF
numeric_features = df.drop(columns=['label', 'text', 'cleaned']).values
X = hstack([X_tfidf, numeric_features])

# 5. Labels
y = (df['label'] == 'spam').astype(int)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

# 7. Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 8. Save the model
joblib.dump((clf, tfidf_vec), "models/spam_classifier.pkl")

# 9. Test prediction
test_texts = [
    "Congratulations! You won a free prize! Call now.",
    "Hey, are we still meeting at 5?",
    "URGENT: You have won $1000 cash!"
    "This is a reminder for our meeting tomorrow.",
    "Free entry in a weekly competition to win FA Cup tickets!",
    "Your appointment is confirmed for next week.",
    "Call me when you get a chance.",
    "Don't forget to submit your assignment by Friday.",
    "Congratulations! You've been selected for a special offer.",
    "This is not spam, just a regular message."
        
]
test_df = pd.DataFrame({"text": test_texts})
test_df['cleaned'] = test_df['text'].apply(lambda t: " ".join(t.lower().split()))

# Add features for test
test_df = add_keyword_flags(test_df)
test_df = extract_basic_features(test_df)
X_test_tfidf = tfidf_vec.transform(test_df['cleaned'])
X_test_all = hstack([X_test_tfidf, test_df.drop(columns=['text', 'cleaned']).values])

# Predict
preds = clf.predict(X_test_all)
for txt, pred in zip(test_texts, preds):
    print(f"{txt} --> {'SPAM' if pred else 'HAM'}")
