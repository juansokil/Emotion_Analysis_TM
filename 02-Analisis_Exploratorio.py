###Copiar Config_Local.py en el mismo directorio y cambiar por el path local abajo
import sys
sys.path.insert(0, '/Users/inesfrias/Documents/Posgrado/Textmining/LeyAborto/TPFinal/scripts')

from Config_Local import directorio_datos 

###Librerias###
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk

#####LEVANTA DATOS#####
base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/unique_tweets.txt", sep='\t', encoding='latin1', low_memory=False)
base_tweets1['horario']=pd.to_datetime(base_tweets1['horario'])  

base_tweets1.columns


###
pd.crosstab(base_tweets1['#SiALaVida'], base_tweets1['#AbortoLegalYa'])
                         
##############ANALISIS EXPLORATORIO######################        
#####Tweets por minutos###TOTAL - POSITIVOS - NEGATIVOS #
base_tweets1.groupby('horario').agg('size').plot(color='black', label='Total')
base_tweets1[base_tweets1['text'].str.contains('#abortolegalta|#mediasancion|#abortolegalseguroygratuito|#quesealey|#abortoseraley|#abortolegalya|#nosotrasdecidimos|#abortolegaloclandestino|#13jabortolegal|#queelabortosealey',regex=True)].groupby('horario').agg('size').plot(color='green', label='A Favor')
base_tweets1[base_tweets1['text'].str.contains('#provida|#salvemoslas2vidas|#salvemoslas2vidas|#noalabortoenargentina|#sialasdosvidas|#argentinaesprovida|#noalaborto|#cuidemoslas2vidas|#sialavida',regex=True)].groupby('horario').agg('size').plot(color='blue', label='En Contra')
#base_tweets1[base_tweets1['text'].str.contains('olmedo', 'Olmedo' , regex=False)  ].groupby('horario').agg('size').plot(color='red', label='Olmedo')
plt.legend(loc='upper left')
plt.xticks(base_tweets1['horario'], rotation=20)
plt.show()     


totales=base_tweets1.groupby('horario').agg('size')                               
favor=base_tweets1[base_tweets1['text'].str.contains('#abortolegalta|#mediasancion|#abortolegalseguroygratuito|#quesealey|#abortoseraley|#abortolegalya|#nosotrasdecidimos|#abortolegaloclandestino|#13jabortolegal|#queelabortosealey',regex=True)].groupby('horario').agg('size')
contra=base_tweets1[base_tweets1['text'].str.contains('#provida|#salvemoslas2vidas|#salvemoslas2vidas|#noalabortoenargentina|#sialasdosvidas|#argentinaesprovida|#noalaborto|#cuidemoslas2vidas|#sialavida',regex=True)].groupby('horario').agg('size')

                                 
###junta los datos en un nuevo df                                 
result = pd.concat([totales,favor,contra],axis=1)
result.columns = ['Total','Favor','Contra']
###calcula el porcentaje relativo sobre el total de tweets
result['pct_favor']=result['Favor']/result['Total']
result['pct_contra']=result['Contra']/result['Total']
####hay un porcentaje importante que no los logra identificar, habria que explorar porque                                 





from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.wv.save_word2vec_format('googlenews.txt')


import spacy
nlp = spacy.load('en',vectors='en_google')

nlp.vocab.load_vectors_from_bin_loc('googlenews.bin')