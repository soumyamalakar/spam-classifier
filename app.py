import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.corpus import stopwords
stopwords.words('english')
import string
stopwords.words('english')

def transform_text(text):
    text=text.lower()
    tokens=nltk.word_tokenize(text)

    y=[]
    for i in tokens:
        if i.isalnum():
            y.append(i)

    text =y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# --- Streamlit UI Design ---
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #4B0082;'>📧 Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Detect whether a message is SPAM or NOT SPAM</p>", unsafe_allow_html=True)
st.markdown("---")

# Input text area
input_sms = st.text_area("Enter the message here:", height=120)

st.markdown("<br>", unsafe_allow_html=True)

# Centered prediction button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button('Predict'):
        if input_sms.strip() == "":
            st.warning("Please enter a message to classify.")
        else:
            # 1. preprocess, vectorize , predict. display
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            if result == 1:
                st.error("🚨 This is SPAM!")
            else:
                st.success("✅ This is NOT SPAM!")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Made with❤️using Streamlit and Python by Soumya</p>", unsafe_allow_html=True)
