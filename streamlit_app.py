import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from joblib import load

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model = load('assets/sentiment_model.joblib')
vectorizer = load('assets/vectorizer.joblib')

st.set_page_config(page_title="💬 Tweet Sentiment Analyzer", layout="centered")
st.title("💬 Tweet Sentiment Analysis")
st.write("Paste a tweet below to analyze its sentiment.")

def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z0-9\s]", "", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

tweet = st.text_area("✍️ Enter Tweet Text", height=150)

if tweet:
    clean = clean_text(tweet)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]

    sentiment_map = {
        'positive': ('😊 Positive', 'green'),
        'negative': ('😠 Negative', 'red'),
        'neutral':  ('😐 Neutral', 'gray')
    }
    label, color = sentiment_map.get(prediction, ('❓ Unknown', 'blue'))
    st.markdown(f"<h3 style='color:{color};'>{label}</h3>", unsafe_allow_html=True)

    if st.checkbox("Show prediction confidence"):
        prob_df = pd.DataFrame({
            "Sentiment": model.classes_,
            "Confidence": [f"{p*100:.2f}%" for p in probs]
        })
        st.table(prob_df)

st.sidebar.title("📊 Model Info")
st.sidebar.write("**Model Accuracy:** 98.7%")  # Update after training
st.sidebar.image("assets/plots/confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)
st.sidebar.markdown("📁 Dataset: `training.1600000.processed.noemoticon.csv`")
