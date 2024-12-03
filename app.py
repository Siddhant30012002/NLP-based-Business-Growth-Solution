# streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
from main import LSTMModel, TextPreprocessor, Vectorizer, LabelEncoding
from keras._tf_keras.keras.models import load_model
import keras

original_conditions = pd.Series(['Birth Control','Depression','Diabetes, Type 2','High Blood Pressure'])
label_encoder = LabelEncoding(original_conditions)

def get_preprocessed_data(your_review):
    dictt = {'review': [your_review]}
    df = pd.DataFrame(dictt, columns=['review'])
    preprocess_the_text = TextPreprocessor(df['review'])
    df["review"] = preprocess_the_text.preprocess()
    vectorizer_1 = Vectorizer(df['review'])
    df['vectorized_review'] = vectorizer_1.vectorize_text(df['review'])
    vectorized_reviews = np.stack(df['vectorized_review'].values)
    reshaped_reviews = vectorized_reviews.reshape((vectorized_reviews.shape[0], 1, vectorized_reviews.shape[1]))
    return reshaped_reviews


LSTM_model = load_model("Updated_tanh_LSTM_Model_1.h5")
st.title(":blue[Disease Categorizer]")
st.write("Enter a medicine review, and the model will predict the related disease.")

with st.form("Review", clear_on_submit=False):
    your_review = st.text_input("Enter Your Medicine Review")
    submit_button = st.form_submit_button("Let me think")

    if submit_button:
        if your_review.strip(): 
            try:
                preprocessed_data = get_preprocessed_data(your_review)
                predictions = LSTM_model.predict(preprocessed_data)
                y_pred = np.argmax(predictions, axis=1)
                disease_name = label_encoder.decode_label(y_pred)
                st.success(f"Predicted Disease: {disease_name[0]}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a valid review!")

    