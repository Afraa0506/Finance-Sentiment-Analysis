import streamlit as st
import pickle
import re
from PIL import Image

with open("../model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

reverse_map = {0:"negative", 1:"neutral", 2:"positive"}

st.set_page_config(
    page_title="Finance Sentiment Analyzer",
    layout="centered",
)

st.markdown("""
    <style>

        /* Background */
        .stApp {
            background-color: #FFD3D5;
        }

        /* Main Title */
        h1 {
            color: #540863 !important;
            text-align: center;
            font-weight: 800;
        }

        /* Subtext */
        .subtext {
            color: #92487A;
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: 500;
        }

        /* Textarea Label */
        label {
            color: #92487A !important;
            font-size: 18px;
        }

        /* Button Styling */
        div.stButton > button {
            background-color: white;
            color: #92487A !important;
            border: 2px solid #E49BA6;
            padding: 0.6rem 1.4rem;
            border-radius: 10px;
            font-size: 17px;
            font-weight: 700;
        }

        div.stButton > button:hover {
            background-color: #E49BA6;
            color: white !important;
        }

    </style>
""", unsafe_allow_html=True)

try:
    image = Image.open("../assets/stock_market.jpg")
    st.image(image, width=700)
except:
    pass  

st.title("Finance Sentiment Analysis")

st.markdown(
    '<p class="subtext">Analyze financial statements, market news, and stock-related sentences.</p>',
    unsafe_allow_html=True
)

user_input = st.text_area("Enter financial text:", height=150)

if st.button("Analyze Statement"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        sentiment = reverse_map[pred]

        if sentiment == "positive":
            st.success(f"Sentiment: **{sentiment.upper()}** 🟢")
        elif sentiment == "neutral":
            st.info(f"Sentiment: **{sentiment.upper()}** ⚪")
        else:
            st.error(f"Sentiment: **{sentiment.upper()}** 🔴")
