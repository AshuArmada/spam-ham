import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOP = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)           # Remove emails
    text = re.sub(r'[^a-z\s]', ' ', text)          # Remove punctuation/numbers
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP]
    return ' '.join(tokens)

def load_and_clean(path):
    df = pd.read_csv(path, sep='\t', names=['label', 'text'])
    df['cleaned'] = df['text'].apply(clean_text)
    return df
