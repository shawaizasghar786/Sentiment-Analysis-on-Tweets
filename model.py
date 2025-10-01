from sklearn.linear_model import LogisticRegression
from joblib import dump
import os

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    os.makedirs('assets', exist_ok=True)
    dump(model, 'assets/sentiment_model.joblib')
    return model
