
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



base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/unique_tweets.txt", sep='\t', encoding='latin1', low_memory=False)


base_tweets1['text']=base_tweets1['text'].str.lower()  

base_tweets1['text_no_ref'] = base_tweets1['text']. \
    apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('@')]))
base_tweets1['text_no_ref'] = base_tweets1['text_no_ref'].\
    apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('http')]))
base_tweets1['text_no_ref'] = base_tweets1['text_no_ref'].\
    apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('rt')]))



###subset bases####
########################FAVOR#########################
#favor=base_tweets1[base_tweets1['text'].str.contains('#abortolegalta|#mediasancion|#abortolegalseguroygratuito|#quesealey|#abortoseraley|#abortolegalya|#nosotrasdecidimos|#abortolegaloclandestino|#13jabortolegal|#queelabortosealey',regex=True)]
                   
####### TGOKENIZAR####
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

corpus = list(sent_to_words(base_tweets1['text_punct']))
print(corpus[:1])

# train model
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


print ("aborto-legal similarity:",model.wv.n_similarity(["aborto"], ["legal"]))
print ("aborto-vida similarity:",model.wv.n_similarity(["aborto"], ["vida"]))

w2="vida"
model.wv.most_similar (positive=w2)


target_word="aborto"
aborto = words
positiva = []
for word in aborto:
    positiva.append(model.wv.n_similarity([target_word], [word]))
    
pd.DataFrame(positiva,index = aborto,columns=[target_word]).plot(kind="bar",figsize=(15,5), fontsize=8)

w1="aborto"
model.wv.most_similar (positive=w1)


target_word="vida"
vida = words
negativa = []
for word in vida:
    negativa.append(model.wv.n_similarity([target_word], [word]))
    
pd.DataFrame(negativa,index = barrios,columns=[target_word]).plot(kind="bar",figsize=(15,5), fontsize=8)

w2="vida"
model.wv.most_similar (positive=w2)


palabras = words
# Armo una matriz de distancias
distancias=np.zeros((len(words),len(words))) #matriz cuadrada
for i,ti in enumerate(words):
    for j,tj in enumerate(words):
        distancias[i,j] = abs(1-model.wv.similarity(ti,tj))
print (distancias.shape)
distancias





palabras = words
# Armo una matriz de distancias
distancias=np.zeros((len(words),len(words))) #matriz cuadrada
for i,ti in enumerate(words):
    for j,tj in enumerate(words):
        distancias[i,j] = abs(1-model.wv.similarity(ti,tj))
print (distancias.shape)
distancias

# Reduccion de la dimensionalidad y visualizacion 
from sklearn.manifold import MDS
from sklearn.manifold import TSNE 
def visualize_embeddings(distancias,palabras,colores,perplexity):
    plt.figure(figsize=(10,10))
    # Reduccion de la dimensionalidad y visualizacion 
    mds = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=123,
                       dissimilarity="precomputed", n_jobs=4)
    Y = mds.fit(distancias).embedding_
    plt.subplot(1,2,1)
    plt.scatter(Y[:, 0], Y[:, 1],color="black",s=3)
    for label, x, y, color in zip(palabras, Y[:, 0], Y[:, 1],colores):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',size=16)
    plt.title("MDS")
    # Reduccion de la dimensionalidad y visualizacion 
    tsne = TSNE(n_components=2,metric="precomputed",learning_rate=1000, random_state=123,perplexity=perplexity)
    np.set_printoptions(suppress=True)
    plt.subplot(1,2,2)
    Y = tsne.fit_transform(distancias)
    plt.scatter(Y[:, 0], Y[:, 1],color="black",s=3)
    for label, x, y, color in zip(palabras, Y[:, 0], Y[:, 1],colores):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',size=16)
    plt.title("TSNE")


visualize_embeddings(distancias,palabras,colores,perplexity=2)




####LOAD GOOGLE NEWS
from gensim.models import KeyedVectors
# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)



####LOAD GLOVE###
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)








