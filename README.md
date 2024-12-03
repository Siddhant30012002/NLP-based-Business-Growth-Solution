# NLP-based Business Growth Solution
## Project Overview

This NLP-based Business Growth Solution utilizes Natural Language Processing (NLP) to classify pharmaceutical customer reviews into categories based on medical conditions, such as diabetes, high blood pressure, asthma and more. The core challenge lies in the fact that patient reviews focus on discussing the drug, not the condition it belongs to. Our NLP solution addresses this by processing drug names and categorizing reviews accordingly. This solution provides business analysts with valuable insights into customer feedback, 
production trends, and growth opportunities.

## Problem Statement
In the pharmaceutical industry, analyzing customer reviews is crucial to understanding market demand and improving product offerings. However, the reviews typically mention the drug name without directly stating the medical condition it treats. Our solution uses NLP 
to infer the medical category from the drug name, thus automating the categorization of reviews and aiding in trend analysis for business growth.

## Features
1. **Automated Classification**: NLP is used to categorize reviews based on medical conditions, such as diabetes, high blood pressure, and asthma.
2. **Drug Recognition**: The solution identifies the drug mentioned in the review, which is crucial for inferring the relevant medical condition.
3. **Business Insights**: The classified reviews provide businesses with actionable insights into customer sentiment and emerging trends.
4. **Data Preprocessing**: The reviews undergo multiple preprocessing steps to enhance classification accuracy.
5. **Model**: Trained using Deep learning techniques (LSTM) for multi-class classification.

## Technologies Used
1. Python
2. Pandas (Data manipulation)
3. NLTK (Text preprocessing)
4. Scikit-learn (Machine learning models)
5. TensorFlow/Keras (Deep learning models)
6. Matplotlib/Seaborn (Data visualization)
7. numpy
8. regex
9. textblob
10. gensim
11. stramlit
12. string

## Dataset
The dataset consists of customer reviews and the associated medical condition labels. The data is preprocessed as follows:

1. **Text preprocessing**: Removing stopwords, HTML tags, URLs, lemmatization, vectorization and more. (connect with me to get te code for preprocessing)
2. **Drug name identification**: Recognizing the drug mentioned in reviews to infer the medical condition.


## How It Works
1. **Data Preprocessing**: Reviews are cleaned and tokenized, removing stop words, HTML tags, URLs, and applying lemmatization.
2. **Drug Recognition**: The NLP model identifies the drug name and other keywords to infer the corresponding medical condition.
3. **Model Training**: A deep learning model is trained to classify reviews into medical categories.
4. **Predictions**: The trained model can classify new reviews into predefined categories.
5. **Business Insights**: Categorized reviews help businesses identify production trends and customer sentiments.

## How to Use
1. Clone this repository.
2. Preprocess the reviews(connect with me to get te code for preprocessing)
3. Train the model 
4. Make predictions 
5. Analyze business insights and trends from the categorized reviews.

## Conclusion
This NLP solution automates the process of categorizing pharmaceutical customer reviews, enabling businesses to gain valuable insights into market trends, customer feedback, and potential areas of improvement. It simplifies the workflow of analyzing customer sentiment and accelerates business growth by providing actionable data.

