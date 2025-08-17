import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def add_keyword_flags(df, keywords=None):
    if keywords is None:
        keywords = ['free', 'win', 'urgent', 'call now', 'prize', 'congratulations']
    for kw in keywords:
        df[f'has_{kw.replace(" ", "_")}'] = df['text'].str.lower().str.contains(
            rf'\b{re.escape(kw)}\b', regex=True
        ).astype(int)
    return df

def extract_basic_features(df):
    df['char_count']       = df['text'].str.len()
    df['word_count']       = df['text'].str.split().apply(len)
    df['avg_word_len']     = df['char_count'] / df['word_count'].replace(0, 1)
    df['num_exclaims']     = df['text'].str.count('!')
    df['num_digits']       = df['text'].str.count(r'\d')
    df['num_uppercase']    = df['text'].str.count(r'[A-Z]')
    df['special_char_pct'] = df['text'].str.count(r'[^A-Za-z0-9 ]') / df['char_count'].replace(0, 1)
    df['num_links']        = df['text'].str.count(r'http[s]?://')
    df['num_emails']       = df['text'].str.count(r'\S+@\S+')
    return df

def tfidf_vectorize(corpus, max_features=1000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                          stop_words='english', min_df=2)
    X = vec.fit_transform(corpus)
    return X, vec
