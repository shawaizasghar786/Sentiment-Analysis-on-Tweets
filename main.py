from src.data_loader import load_data
from src.preprocessing import preprocess
from src.vectorizer import vectorize_text
from src.model import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = r'D:\Coding\Sentiment-Analysis-on-Tweets\training.1600000.processed.noemoticon.csv'
df = load_data(path)
df = preprocess(df)

X, vectorizer = vectorize_text(df['clean_text'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(X_train, y_train)
y_pred = evaluate_model(model, X_test, y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")
plot_confusion_matrix(y_test, y_pred, labels=y.unique())
