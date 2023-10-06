ML project for restaurant review served with Streamlit.

Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes you can build and deploy powerful data apps - so let's get started!

### Requirements
You must have Pandas, NLTK Tools and Streamlit installed.
```
pip install streamlit
```
```
pip install -U nltk  
```
### Running the project
1. Ensure that you are in the project home directory. Run the notebook : "Sentimental Analysis- Restaurant Review" first

This would create a serialized versions of our models into files "Sentiment_Prediction_model" & "tfidf-transform"

2. Run app.py using below command to start Streamlit web framework
```
streamlit run app.py
```
By default, Streamlit will run on port 8502

3. Navigate to URL http://localhost:8502/

You should be able to view the homepage.

Enter the restaurant review in form of a statement like "The food here was not good" in the text field and hit Submit.

If everything goes well, you should  be able to see the review

```
Hit Star if you like this project:
```

