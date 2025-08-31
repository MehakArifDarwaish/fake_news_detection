# app.py
import streamlit as st
import joblib
import re
import string
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run only first time)
# import nltk
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("punkt")

import nltk
from nltk.data import find

def safe_download(resource):
    try:
        find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])

safe_download("tokenizers/punkt")
safe_download("tokenizers/punkt_tab")

# Load saved model and vectorizer
model = joblib.load("best_model.pkl")       # model
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # TF-IDF vectorizer

# Text Preprocessing Function
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Remove URLs, numbers, punctuation
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Lowercase and tokenize
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Paste a news article below and the model will predict whether it is **Fake** or **Real**.")

# User input
user_input = st.text_area("Enter News Article Text", height=250)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Preprocess
        cleaned_text = clean_text(user_input)

        # Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]

        # Display results
        if prediction == "FAKE":
            st.error(f"🚨 News is **Fake** (Confidence: {probability[0]*100:.2f}%)")
        else:
            st.success(f"✅ News is **Real** (Confidence: {probability[1]*100:.2f}%)")

        # Show probability chart
        st.write("### Prediction Confidence")
        st.bar_chart({"Fake": [probability[0]], "Real": [probability[1]]})

st.markdown("---")
st.markdown("🔍 **Built with Gradient Boosting with accuracy 0.9960 + TF-IDF**")
st.markdown(" **Deployed by : Hafiza Mehak Arif**")