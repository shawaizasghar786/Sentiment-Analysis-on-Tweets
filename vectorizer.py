from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

def vectorize_text(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    dump(vectorizer, 'assets/vectorizer.joblib')
    return X, vectorizer
