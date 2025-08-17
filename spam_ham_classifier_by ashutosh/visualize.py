def plot_distributions():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    from scr.preprocess import load_and_clean
    from scr.features import add_keyword_flags, extract_basic_features
    from sklearn.metrics import confusion_matrix
    import joblib
    from scipy.sparse import hstack
    import os

    os.makedirs("visuals", exist_ok=True)



 #code done by ashutosh


 
    # Load model, vectorizer, and keywords from training
    clf, tfidf_vec, keywords = joblib.load("models/spam_classifier.pkl")

    # Load dataset
    df = load_and_clean("data/raw/SMSSpamCollection")
    df = add_keyword_flags(df, keywords)  # Use same keywords from training
    df = extract_basic_features(df)

    # Vectorize using saved TF-IDF vectorizer
    X_tfidf = tfidf_vec.transform(df['cleaned'])
    numeric_features = df.drop(columns=['label', 'text', 'cleaned']).values
    X_all = hstack([X_tfidf, numeric_features])
    y_true = (df['label'] == 'spam').astype(int)

    # Spam vs Ham count
    df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Spam vs Ham Distribution")
    plt.savefig("visuals/spam_ham_dist.png")
    plt.clf()

    # Word count distribution
    sns.histplot(df[df['label']=='spam']['word_count'], bins=30, color='red', label='Spam', alpha=0.6)
    sns.histplot(df[df['label']=='ham']['word_count'], bins=30, color='blue', label='Ham', alpha=0.6)
    plt.legend()
    plt.title("Word Count Distribution")
    plt.savefig("visuals/word_count_dist.png")
    plt.clf()

    # Keyword flags
    keyword_cols = [c for c in df.columns if c.startswith("has_")]
    if keyword_cols:
        df.groupby('label')[keyword_cols].sum().T.plot(kind='bar', figsize=(8,5))
        plt.title("Keyword Occurrences")
        plt.savefig("visuals/keyword_flags.png")
        plt.clf()

    # WordCloud for spam
    spam_text = ' '.join(df[df['label'] == 'spam']['cleaned'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
    wc.to_file("visuals/spam_wordcloud.png")

    # 📊 Confusion Matrix Heatmap
    y_pred = clf.predict(X_all)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("visuals/confusion_matrix.png")
    plt.clf()

    print("✅ Visuals saved in 'visuals/'")

if __name__ == "__main__":
    plot_distributions()
