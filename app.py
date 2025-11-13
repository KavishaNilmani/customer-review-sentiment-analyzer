import re
import nltk
import pandas as pd
import streamlit as st
import scipy.sparse as sp
from joblib import load
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# load model amd vectorizers
bundle = load("vectorizers.joblib")
word_vectorizer = bundle["word"]
char_vectorizer = bundle["char"]
model = load("sentiment_model.joblib")

# preprocessing 
def clean_text(text):
    text = text.lower()
    text = str(text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9.,!?\' ]", " ", text)
    return text

negation_words = {
    "no","not","nor","never","n't","don't","dont","cannot","can't",
    "hardly","barely","scarcely","won't","wouldn't","shouldn't","isn't",
    "wasn't","aren't","weren't","couldn't","didn't" 
}
base_stopwords = set(stopwords.words("english"))
stop_words = base_stopwords - negation_words

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

lemmatizer = WordNetLemmatizer()
def _map_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def lemmatize_text(text):
    tokens = text.split()
    pos_tags = nltk.pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(tok, _map_pos(pos)) for tok, pos in pos_tags]
    return " ".join(lemmas)

def prepare_text(text):
    t = clean_text(text)
    t = lemmatize_text(t)
    t = remove_stopwords(t)
    return t

def vectorize_text(text):
    t = prepare_text(text)
    xw = word_vectorizer.transform([t])
    xc = char_vectorizer.transform([t])
    return sp.hstack([xw, xc]).tocsr(), t

CONF_TRESH = 0.40
MIN_TOKENS = 2

# UI
st.set_page_config(page_title="Customer Review Sentiment", layout="centered")
st.title("Customer Review Sentiment Analyzer")
st.caption("Type a review, see predicted sentiment and class probabilities.")

review = st.text_area("Enter a review", height=120, placeholder="Type a customer review here...")
if st.button("Analyze") and review.strip():
    X, processed = vectorize_text(review)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        labels = list(model.classes_)
        pred = labels[int(probs.argmax())]
        max_p = float(probs.max())

        # neutral only when both confidence is low
        if (max_p < CONF_TRESH) and (len(processed.split()) < MIN_TOKENS):
            pred = "neutral"
        # if max_p < CONF_TRESH or len(processed.split()) < MIN_TOKENS:
        #     pred = "neutral"
        df = pd.DataFrame({"class": labels, "probability": probs}).sort_values("probability", ascending=False)
        st.subheader(f"Predicted: {pred}")
        st.bar_chart(df.set_index("class"))
    else:
        pred = model.predict(X)[0]
        st.subheader(f"Predicted: {pred}")
    with st.expander("Show preprocessed text"):
        st.code(processed, language="text")

