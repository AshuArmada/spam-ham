import sys
import joblib
import pandas as pd
from scipy.sparse import hstack
from scr.features import add_keyword_flags, extract_basic_features

# Load model and vectorizer
clf, tfidf_vec, keywords = joblib.load("models/spam_classifier.pkl")

# Get input
text = sys.argv[1] if len(sys.argv) > 1 else input("Enter a message: ")


 #code done by ashutosh
# Prepare DataFrame
df = pd.DataFrame({"text": [text], "cleaned": [" ".join(text.lower().split())]})
df = add_keyword_flags(df, keywords)
df = extract_basic_features(df)

# Vectorize
X_tfidf = tfidf_vec.transform(df['cleaned'])
numeric_features = df.drop(columns=['text', 'cleaned']).values
X_all = hstack([X_tfidf, numeric_features])

# Predict
pred = clf.predict(X_all)[0]
print("SPAM" if pred else "HAM")
