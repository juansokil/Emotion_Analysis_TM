
#%matplotlib inline 
#%load_ext autoreload
#%autoreload 2
from collections import defaultdict
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd   
from gensim import corpora, models, similarities, matutils
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from gensim.matutils import cossim
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
import pickle
import random
import sys
from tqdm import tqdm
from sklearn import manifold


from gensim.models import Word2Vec
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


####A FAVOR DE LA LEGALIZACION####
base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/unique_tweets.txt", sep='\t', encoding='latin1', low_memory=False)
base_tweets1=base_tweets1[base_tweets1['text'].str.contains('#abortolegalta|#mediasancion|#abortolegalseguroygratuito|#quesealey|#abortoseraley|#abortolegalya|#nosotrasdecidimos|#abortolegaloclandestino|#13jabortolegal|#queelabortosealey',regex=True)]
                   
####### TGOKENIZAR####
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

corpus = list(sent_to_words(base_tweets1['text_punct']))
print(corpus[:1])

#################A FAVOR
model = Word2Vec(corpus, min_count=3070)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

w1="aborto"
model.wv.most_similar (positive=w1)




####EN CONTRA DE LA LEGALIZACION####
base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/unique_tweets.txt", sep='\t', encoding='latin1', low_memory=False)
base_tweets1=base_tweets1[base_tweets1['text'].str.contains('#provida|#salvemoslas2vidas|#salvemoslas2vidas|#noalabortoenargentina|#sialasdosvidas|#argentinaesprovida|#noalaborto|#cuidemoslas2vidas|#sialavida',regex=True)]

####### TGOKENIZAR####
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

corpus = list(sent_to_words(base_tweets1['text_punct']))
print(corpus[:1])

###EN CONTRA
model = Word2Vec(corpus, min_count=597)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

w2="vida"
model.wv.most_similar (positive=w2)







####LOAD GOOGLE NEWS### - EJEMPLO - NO LO USE PARA NADA
from gensim.models import KeyedVectors
# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)


####LOAD GLOVE### - EJEMPLO - NO LO USE PARA NADA
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)








