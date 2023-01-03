# Importing libraries
import numpy as np
import pandas as pd
import pickle
import nltk
import re
from nltk.stem.porter import PorterStemmer
import streamlit as st

st.title("Restaurant Review's Sentiment Analyzer")
st.markdown("*A Machine Learning Web App built with Streamlit*")
st.write("")

#Taking input from user
sample_message = st.text_area('Enter your review here...', height=250)

# Load the Naive Bayes model and CounterVectorizer object from the disk
model = pickle.load(open('Sentiment_Prediction_model', 'rb'))
vectorizer = pickle.load(open('tfidf-transform', 'rb'))

# we are removing the words from the stop words list: 'no', 'nor', 'not',isn't,"doesn't", "won't"
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn','hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn','ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won','wouldn', "wouldn't"])

# Prediction function
def predict_sentiment(sample_review):
    sample_review = re.sub('[^a-zA-Z]', ' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review = sample_review.split()
    
    ps = PorterStemmer()
    sample_review = [ps.stem(word) for word in sample_review if word not in stopwords]
    sample_review = ' '.join(sample_review)
    
    temp = vectorizer.transform([sample_review]).toarray()
    return model.predict(temp)[0]

# Submit button

if st.button('Submit'):
    result = predict_sentiment(sample_message)

    if result == 0:
        st.write('*Negative Review*')
    else:
        st.write('*Positive Review*')        



