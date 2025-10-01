import pandas as pd

def load_data(path):
    df = pd.read_csv(path, encoding='latin-1', header=None)
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = df[['text', 'target']]
    df['sentiment'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
    return df[['text', 'sentiment']]
