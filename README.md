# SMS-Spam-Classifier
End to end code for the email spam classifier project


This project is an Email/SMS Spam Classifier built using the Streamlit framework. It aims to classify incoming messages as either spam or not spam. The classifier is trained on a dataset of labeled messages using a machine learning model, which is loaded from a pickle file. The input message is preprocessed by transforming the text, removing stopwords, punctuation, and applying stemming. Then, the transformed message is vectorized using a TF-IDF vectorizer. Finally, the model predicts the class of the message (spam or not spam), and the result is displayed using Streamlit's user interface.
