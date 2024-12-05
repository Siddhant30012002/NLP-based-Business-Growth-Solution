import keras._tf_keras
import pandas as pd
import numpy as np
import pickle
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
import re
import string
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import Word2Vec
print(gensim.__version__)
import sklearn 
import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM,Dense,Dropout
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.regularizers import L2


file_path = "C:/Users/HP/Desktop/my_nlp_project/drugsComTrain.tsv"


class DataFrame_Creator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path, sep='\t')
        return self.data
    
class TextPreprocessor:
    
    def __init__(self,review):
        self.review = review
        self.review = self.review.str.lower()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
    
    def remove_html_tags(self):
        self.review = self.review.apply(lambda x: re.sub(r"<[^>]*>", '', x))
        return self.review
    
    def remove_url(self):
        self.review = self.review.apply(lambda x: re.sub(r"http\S+|www\S+",'',x))
        return self.review
    
    def punctuations(self):
        self.review = self.review.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
        return self.review
    
    def remove_stopwords(self):
        def remove_stopwords_from_review(review):
            words = review.split()
            return ' '.join(word for word in words if word not in self.stop_words)
        self.review = self.review.apply(remove_stopwords_from_review)
        return self.review
    
    def lemmatization(self):
        self.review = self.review.apply(lambda x:' '.join([self.lemmatizer.lemmatize(word) for word in x.split()]))
        return self.review
    
    def tokenizer(self):
        self.review = self.review.apply(lambda x:x.split())
        return self.review
    
    def preprocess(self):
        self.remove_html_tags()
        self.remove_url()
        self.punctuations()
        self.remove_stopwords() 
        self.lemmatization()
        self.tokenizer()
        return self.review
    
class Vectorizer:
    def __init__(self,tokenized_review, vector_size=100, window=5, min_count=1):
        self.model = gensim.models.Word2Vec(sentences=tokenized_review, vector_size=vector_size, window=window, min_count=min_count)

    def vectorize_text(self,tokenized_review):
        vectors = []
        for tokens in tokenized_review:
            word_vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if word_vectors:
                vectors.append(sum(word_vectors) / len(word_vectors))  
            else:
                vectors.append(np.zeros(self.model.vector_size))  
        print("Vectorized sample shape:", np.array(vectors).shape)
        return vectors
    
class LabelEncoding:
    def __init__(self,condition):
        self.condition = condition.str.lower()
        self.le = sklearn.preprocessing.LabelEncoder()
        self.encoded_condition = self.le.fit_transform(condition)
    
    def get_encoded_lable(self):
        return self.encoded_condition
    
    def decode_label(self, encoded_labels):
        return self.le.inverse_transform(encoded_labels)
    
class DataSplitter:
    def __init__(self,data,test_size=0.20,random_state=25): 
        self.X = data[0]
        self.y = data[1]
        self.test_size = test_size
        self.random_state = random_state
        self.X_train,self.X_test,self.y_train,self.y_test=sklearn.model_selection.train_test_split(self.X,self.y,
                                                                                                   test_size=self.test_size,
                                                                                                   random_state=self.random_state)

    def get_train_data(self):
        return self.X_train,self.y_train
    
    def get_test_data(self):
        return self.X_test,self.y_test
    

class ReshapeData:
    def __init__(self, X_train_reshape, X_test_reshape):
        self.X_train_reshape = np.array(X_train_reshape.tolist()).astype('float32')
        self.X_test_reshape = np.array(X_test_reshape.tolist()).astype('float32')
        self.X_train_reshaped = self.X_train_reshape.reshape(len(self.X_train_reshape), 1, 100)
        self.X_test_reshaped = self.X_test_reshape.reshape(len(self.X_test_reshape), 1, 100)
    def get_reshaped_data(self):
        return self.X_train_reshaped, self.X_test_reshaped
    
class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=input_shape, activation='tanh',return_sequences=True),L2(0.01))
        self.model.add(LSTM(32, activation='tanh'))
        self.model.add(Dense(4, activation='softmax'))
        #self.model.add(Dropout(0.3))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        return history
























































