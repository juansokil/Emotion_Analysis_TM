###Copiar Config_Local.py en el mismo directorio y cambiar por el path local abajo
import sys
sys.path.insert(0, '/Users/inesfrias/Documents/Posgrado/Textmining/LeyAborto/TPFinal/scripts')

from Config_Local import directorio_datos 


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

###subset bases####
########################FAVOR#########################
favor=base_tweets1[base_tweets1['text'].str.contains('#abortolegalta|#mediasancion|#abortolegalseguroygratuito|#quesealey|#abortoseraley|#abortolegalya|#nosotrasdecidimos|#abortolegaloclandestino|#13jabortolegal|#queelabortosealey',regex=True)]
                   
####### TGOKENIZAR####
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

corpus_favor = list(sent_to_words(favor['text_punct']))
print(corpus_favor[:1])

# train model
model = Word2Vec(corpus_favor, min_count=400)
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


w1="aborto"
model.wv.most_similar (positive=w1)
w2="vida"
model.wv.most_similar (positive=w2)




                                   
contra=base_tweets1[base_tweets1['text'].str.contains('#provida|#salvemoslas2vidas|#salvemoslas2vidas|#noalabortoenargentina|#sialasdosvidas|#argentinaesprovida|#noalaborto|#cuidemoslas2vidas|#sialavida',regex=True)]

####### TGOKENIZAR####
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

corpus_contra = list(sent_to_words(contra['text_punct']))
print(corpus_contra[:1])

# train model
model = Word2Vec(corpus_contra, min_count=150)


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


w1="aborto"
model.wv.most_similar (positive=w1)
w2="vida"
model.wv.most_similar (positive=w2)


    




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












print ("mujer-cocina similarity:",w2v_model.wv.n_similarity(["mujer"], ["cocina"]))
print ("hombre-cocina similarity:",w2v_model.wv.n_similarity(["hombre"], ["cocina"]) )
print ("\n")
print ("mujer-esposa similarity:",w2v_model.wv.n_similarity(["mujer"], ["esposa"]) )
print ("hombre-esposo similarity:",w2v_model.wv.n_similarity(["hombre"], ["esposo"]) )
print("\n")
print ("mujer-hijos similarity:",w2v_model.wv.n_similarity(["mujer"], ["hijos"]) )
print ("hombre-hijos similarity:",w2v_model.wv.n_similarity(["hombre"], ["hijos"]) )


w2v_model.most_similar(positive=["biología"], negative=[], topn=25)

w2v_model.most_similar(positive=["computación"], negative=[], topn=25)

target_word="crimen"
barrios = ["belgrano","caballito","ortúzar","palermo","recoleta","núñez","lugano","pompeya","martelli","flores","barracas","soldati","cañitas"]
crimen = []
for word in barrios:
    crimen.append(w2v_model.wv.n_similarity([target_word], [word]))
    
pd.DataFrame(crimen,index = barrios,columns=[target_word]).sort_values(by=target_word).plot(kind="bar",figsize=(15,5), fontsize=20)

p_robos = ["robos","armas","asesinato","ladrones","hurto","asalto"]
p_ciencias = ["biología","química","matemática","filosofía","psicología","ciencia","ingeniería"]
p_tiempo = ["lluvioso","soleado","calor","nublado","nieve","tormenta"]
p_paises = ["suiza","suecia","francia","holanda","australia","perú","bolivia","paraguay","uruguay","brasil","colombia"]
p_comida = ["pan","fideos","galletitas","queso","pizza","cerveza","vino"]
p_tecno = ["tecnología","computadora","internet","web","hackers","monitor","mouse"]
p_hogar = ["cocina","baño","comedor","sillones","armario","sillas","mesas","vajilla"]
palabras = p_robos + p_ciencias + p_tiempo + p_paises + p_comida+p_tecno+p_hogar
colores = ["black"]*len(p_robos)+["blue"]*len(p_ciencias)+["green"]*len(p_tiempo)+["red"]*len(p_paises) +["purple"]*len(p_comida)+["orange"]*len(p_tecno)+["cyan"]*len(p_hogar) 

# Armo una matriz de distancias
distancias=np.zeros((len(palabras),len(palabras))) #matriz cuadrada
for i,ti in enumerate(palabras):
    for j,tj in enumerate(palabras):
        distancias[i,j] = abs(1-w2v_model.wv.similarity(ti,tj))
print (distancias.shape)
distancias

# Reduccion de la dimensionalidad y visualizacion 
from sklearn.manifold import MDS
from sklearn.manifold import TSNE 
def visualize_embeddings(distancias,palabras,colores,perplexity):
    plt.figure(figsize=(20,10))
    # Reduccion de la dimensionalidad y visualizacion 
    mds = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=123,
                       dissimilarity="precomputed", n_jobs=4)
    Y = mds.fit(distancias).embedding_
    plt.subplot(1,2,1)
    plt.scatter(Y[:, 0], Y[:, 1],color="black",s=3)
    for label, x, y, color in zip(palabras, Y[:, 0], Y[:, 1],colores):
        plt.annotate(label, xy=(x, y), xytext=(0, 0),color=color, textcoords='offset points',size=13)
    plt.title("MDS")
    # Reduccion de la dimensionalidad y visualizacion 
    tsne = TSNE(n_components=2,metric="precomputed",learning_rate=1000, random_state=123,perplexity=perplexity)
    np.set_printoptions(suppress=True)
    plt.subplot(1,2,2)
    Y = tsne.fit_transform(distancias)
    plt.scatter(Y[:, 0], Y[:, 1],color="black",s=3)
    for label, x, y, color in zip(palabras, Y[:, 0], Y[:, 1],colores):
        plt.annotate(label, xy=(x, y), xytext=(0, 0),color=color, textcoords='offset points',size=13)
    plt.title("TSNE")


visualize_embeddings(distancias,palabras,colores,perplexity=4)

    