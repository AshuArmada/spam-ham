def train_model():
    import joblib
    from scipy.sparse import hstack
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from scr.preprocess import load_and_clean
    from scr.features import add_keyword_flags, extract_basic_features, tfidf_vectorize
    from collections import Counter

    # Load data
    df = load_and_clean("data/raw/SMSSpamCollection")

    # Auto-discover top spam words
    spam_words = ' '.join(df[df['label'] == 'spam']['cleaned']).split()
    top_spam_terms = [word for word, _ in Counter(spam_words).most_common(10)]

    # Add features
    df = add_keyword_flags(df, top_spam_terms)
    df = extract_basic_features(df)

    # Vectorize text
    X_tfidf, tfidf_vec = tfidf_vectorize(df['cleaned'])

    # Combine all features
    numeric_features = df.drop(columns=['label', 'text', 'cleaned']).values
    X = hstack([X_tfidf, numeric_features])
    y = (df['label'] == 'spam').astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Train model
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # 🔹 Evaluate model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Model Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model & vectorizer
    joblib.dump((clf, tfidf_vec, top_spam_terms), "models/spam_classifier.pkl")
    print("✅ Model trained and saved!")

if __name__ == "__main__":
    train_model()
