###Librerias###
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime



###Levanta datos
base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/data.tweet_tabulador.txt", sep='\t', encoding='latin1', low_memory=False)
base_tweets1.columns

###ELIMINO UN CASO QUE LO LEVANTO MAL
base_tweets1=base_tweets1.drop(base_tweets1.index[[169761]])
##RESETEO EL INDEX
base_tweets1=base_tweets1.reset_index(drop=True)


###DIA HORA
#dia_hora = []
#for i in range(len(base_tweets1)):
#    dato=datetime.strptime(str(base_tweets1['created_at'][i]),'%a %b %d %H:%M:%S +0000 %Y')
#    dato_format=dato.strftime('%d %b %H:%M');
#    print(dato_format)
#    dia_hora.append(dato_format)

base_tweets1['horario']=pd.to_datetime(base_tweets1['created_at'])
base_tweets1['horario'] = base_tweets1['horario'] - pd.Timedelta('03:00:00') 


##POR AHORA QUEDA PENDIENTE
#localidades_reg_ex = base_tweets1_reduc.groupby('location').agg('size').sort_values(ascending=False)
#nombres_reg_ex = base_tweets1_reduc.groupby('name').agg('size').sort_values(ascending=False)


#####Tweets por minutos###TOTAL - POSITIVOS - NEGATIVOS #
base_tweets1.groupby('horario').agg('size').plot(color='black')
base_tweets1[base_tweets1['#AbortoLegalYa'] | base_tweets1['#AbortoLegalSeguroYGratuito'] | base_tweets1['#AbortoSeraLey'] | base_tweets1['#QueSeaLey'] | base_tweets1['#13JAbortoLegal'] | base_tweets1['#VotenAbortoLegal']  ].groupby('horario').agg('size').plot(color='green')
base_tweets1[base_tweets1['#SalvemosLas2Vidas'] | base_tweets1['#ArgentinaEsProVida']  | base_tweets1['#NoAlAbortoEnArgentina'] | base_tweets1['#SiALaVida'] | base_tweets1['#NoAlAbortoEnArgentina'] ].groupby('horario').agg('size').plot(color='blue')                 
base_tweets1[base_tweets1['text'].str.contains('Carrio', 'carrio', regex=False)].groupby('horario').agg('size').plot(color='yellow')
base_tweets1[base_tweets1['text'].str.contains('olmedo', 'Olmedo' , regex=False)  ].groupby('horario').agg('size').plot(color='red')
plt.xticks(base_tweets1['horario'],rotation = 150)
plt.locator_params(axis='x', nbins=20)
plt.show()     


totales=base_tweets1.groupby('horario').agg('size')
favor=base_tweets1[base_tweets1['#AbortoLegalYa'] | base_tweets1['#AbortoLegalSeguroYGratuito'] | base_tweets1['#AbortoSeraLey'] | base_tweets1['#QueSeaLey'] |base_tweets1['#13JAbortoLegal'] | base_tweets1['#VotenAbortoLegal']  ].groupby('horario').agg('size')
contra=base_tweets1[base_tweets1['#SalvemosLas2Vidas'] | base_tweets1['#ArgentinaEsProVida']  | base_tweets1['#NoAlAborto'] | base_tweets1['#SiALaVida']  | base_tweets1['#NoAlAbortoEnArgentina']  ].groupby('horario').agg('size')
                               
                                 
###junta los datos en un nuevo df                                 
result = pd.concat([totales,favor,contra],axis=1)
result.columns = ['Total','Favor','Contra']
###calcula el porcentaje relativo sobre el total de tweets
result['pct_favor']=result['Favor']/result['Total']
result['pct_contra']=result['Contra']/result['Total']
####hay un porcentaje importante que no los logra identificar, habria que explorar porque                                 

###GRAFICA###
result['pct_favor'].plot(color='green')
result['pct_contra'].plot(color='blue')                                 
plt.xticks(base_tweets1['horario'],rotation = 150)
plt.locator_params(axis='x', nbins=20)
plt.show()





##SUBSET PARA TRABAJAR MAS FACIL
base_tweets1=base_tweets1.head(n=100)


####PREPROCESAMIENTO####
##pasa todo a minuscula
base_tweets1['text']=base_tweets1['text'].str.lower()

###Cargo stopwords
from nltk.corpus import stopwords
stopwords=stopwords.words('spanish')
base_tweets1['text_stopwords'] = base_tweets1['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
print(base_tweets1['text'][2])
print(base_tweets1['text_stopwords'][2])

###Aplica stemmer (No es muy lindo - habria que probar un lematizador)
from nltk.stem import SnowballStemmer
stemmer_spanish = SnowballStemmer('spanish')
base_tweets1['text_stopwords_stem'] = base_tweets1['text_stopwords'].apply(lambda x: ' '.join([stemmer_spanish.stem(word) for word in x.split()]))
print(base_tweets1['text_stopwords'][2])
print(base_tweets1['text_stopwords_stem'][2])



###Arma el tokenizer de twitter 
tw = TweetTokenizer()

##lo aplica a los tweets completos- no la uso - lo hace pero no me sirve para el vectorizer
tokens=base_tweets1['text_stopwords_stem'].apply(tw.tokenize)
se = pd.Series(tokens)
base_tweets1['tokenized'] = pd.DataFrame(se.values)

print(base_tweets1['text_stopwords_stem'][2])
print(base_tweets1['tokenized'][2])



###ARMO EL BAG OF WORDS
tf=base_tweets1['tokenized'].apply(pd.value_counts).fillna(0).astype(int)


###### HASTA ACA ESTA BIEN#####



from sklearn.feature_extraction.text import CountVectorizer
no_features=1000
#Armamos el vectorizer 

vectorizer = CountVectorizer(analyzer = "word",
                #min_df=10, 
                ngram_range=(1,2),
                #tokenizer=tw,
                token_pattern='[a-zA-Z0-9]{3,}',
                 max_features = no_features, tokenizer=tw.tokenize(tipo))

###Aplica el modelo a los datos, en el ejemplo ejecuto una sola fila (para que salga rapido)
tf = vectorizer.fit_transform(base_tweets1['text'].astype(str))    
tf = vectorizer.fit_transform(base_tweets1['text'].apply(tw.tokenize)[:2])

#Genera el vocabulario
tf_feature_names = vectorizer.get_feature_names()
vocab = vectorizer.get_feature_names()
len(vocab)







# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
# Plotting tools
import pyLDAvis.sklearn

###############LDA WITH SKLEARN

# Build LDA Model
lda_model = LatentDirichletAllocation(n_topics=20,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )

lda_output = lda_model.fit_transform(tf)
print(lda_model)  # Model attributes


LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7,
             learning_method='online', learning_offset=10.0,
             max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
             n_components=10, n_jobs=-1, n_topics=20, perp_tol=0.1,
             random_state=100, topic_word_prior=None,
             total_samples=1000000.0, verbose=0)


# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(tf))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(tf))

# See model parameters
pprint(lda_model.get_params())


















#####GRID PARAMETROS OPTIMOS#####

# Define Search Param
search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)



GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1,
             n_topics=None, perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
       


# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))



# Get Log Likelyhoods from Grid Search Output
n_topics = [10, 15, 20, 25, 30]
log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.5]
log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.7]
log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.9]

# Show graph
plt.figure(figsize=(12, 8))
plt.plot(n_topics, log_likelyhoods_5, label='0.5')
plt.plot(n_topics, log_likelyhoods_7, label='0.7')
plt.plot(n_topics, log_likelyhoods_9, label='0.9')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()







http://cfss.uchicago.edu/fall2016/webdata_lab07b.html


##https://stackoverflow.com/questions/44173624/how-to-nltk-word-tokenize-to-a-pandas-dataframe-for-twitter-data