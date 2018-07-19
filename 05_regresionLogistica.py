# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:44:27 2018

@author: Juan
"""
from sklearn.decomposition import PCA
from matplotlib import pyplot

import pandas as pd
import numpy as np
import re, nltk, spacy, gensim
import spacy
nlp = spacy.load('en_core_web_sm')


# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint



base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/unique_tweets.txt", sep='\t', encoding='latin1', low_memory=False)


from sklearn.utils import shuffle
base_tweets1 = shuffle(base_tweets1)                   



base_tweets1['text']=base_tweets1['text'].str.lower()  

base_tweets1['text_no_ref'] = base_tweets1['text']. \
    apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('@')]))
base_tweets1['text_no_ref'] = base_tweets1['text_no_ref'].\
    apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('http')]))
base_tweets1['text_no_ref'] = base_tweets1['text_no_ref'].\
    apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('rt')]))




####ARMADO DEL BAG OF WORDS######
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from __future__ import print_function
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer



#Bag of words
#Genera la matriz bag-of-words - en el index pone los articulos y en la columna las palabras
#bagofwords el que vale - aca se puede definir el porcentaje de apariciones que pueden tener las palabras
no_features=5000
tf_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, min_df=0.001, max_df=0.70, ngram_range=(1,2), stop_words = 'english', max_features = no_features)
tf = tf_vectorizer.fit_transform(base_tweets1['text_no_ref'])
bagofwords= pd.DataFrame(tf.toarray(),columns=tf_vectorizer.get_feature_names())
bagofwords.shape
#Genera el vocabulario
tf_feature_names = tf_vectorizer.get_feature_names()
vocab = tf_vectorizer.get_feature_names()
len(vocab)


###ARMA TARGET A favor es el TRUE####
bagofwords['target']=base_tweets1['text'].str.contains('#abortolegalta|#mediasancion|#abortolegalseguroygratuito|#quesealey|#abortoseraley|#abortolegalya|#nosotrasdecidimos|#abortolegaloclandestino|#13jabortolegal|#queelabortosealey',regex=True)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics      


from sklearn.model_selection import train_test_split
train, test = train_test_split(bagofwords, test_size=0.2)
#train_target=train['text'].str.contains('#abortolegalta|#mediasancion|#abortolegalseguroygratuito|#quesealey|#abortoseraley|#abortolegalya|#nosotrasdecidimos|#abortolegaloclandestino|#13jabortolegal|#queelabortosealey',regex=True)
#test_target=test['text'].str.contains('#abortolegalta|#mediasancion|#abortolegalseguroygratuito|#quesealey|#abortoseraley|#abortolegalya|#nosotrasdecidimos|#abortolegaloclandestino|#13jabortolegal|#queelabortosealey',regex=True)


          
# MODELO
#---------------------------------------------------------------------------------------------
modelo_lr = LogisticRegression()
modelo_lr.fit(X=train.loc[:, train.columns != 'target'],y=train['target'])          
          
# PREDICCION
#---------------------------------------------------------------------------------------------
prediccion = modelo_lr.predict(test.loc[:, test.columns != 'target'])

# METRICAS
#---------------------------------------------------------------------------------------------
print(metrics.classification_report(y_true=test['target'], y_pred=prediccion))

          
          