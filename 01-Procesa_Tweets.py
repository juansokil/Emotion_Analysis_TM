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
 
###elimino segundos
base_tweets1['horario'] = base_tweets1['horario'].apply(lambda t: t.replace(second=0))

##SUBSET PARA TRABAJAR MAS FACIL
###Quitar los duplicados, mantener el ultimo##
unique_tweets=base_tweets1.groupby(['text']).min()
unique_tweets=unique_tweets.reset_index()

###convierte el df a csv

unique_tweets.columns

unique_tweets.drop(columns=['in_reply_to_user_id_str', 'lang',
                            'favorited', 'truncated',
       'id_str','in_reply_to_status_id_str','statuses_count',
       'followers_count', 'favourites_count', 'protected', 'time_zone','user_lang'])

unique_tweets.to_csv('./unique_tweets.txt',  header=True, sep='\t', encoding='latin1')












#####LEVANTAR DATOS#####
base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/unique_tweets.txt", sep='\t', encoding='latin1', low_memory=False)
base_tweets1['horario']=pd.to_datetime(base_tweets1['horario'])
base_tweets1['text']=base_tweets1['text'].str.lower()  
        
        
        
        
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




##SUBSET PARA TRABAJAR MAS FACIL
base_tweets1=base_tweets1.head(n=10000)


####PREPROCESAMIENTO####

###Cargo stopwords

base_tweets1['text_punct'] = base_tweets1['text'].str.replace('[^\w\s]','')

from nltk.corpus import stopwords
stopwords=stopwords.words('spanish')
base_tweets1['text_stopwords'] = base_tweets1['text_punct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))


###Aplica stemmer (No es muy lindo - habria que probar un lematizador)
from nltk.stem import SnowballStemmer
stemmer_spanish = SnowballStemmer('spanish')
base_tweets1['text_stopwords_stem'] = base_tweets1['text_stopwords'].apply(lambda x: ' '.join([stemmer_spanish.stem(word) for word in x.split()]))


###Elimina usuarios, paginas y rt
base_tweets1['text_stopwords_stem2'] = base_tweets1['text_stopwords_stem'].apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('@')]))
base_tweets1['text_stopwords_stem2'] = base_tweets1['text_stopwords_stem2'].apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('http')]))
base_tweets1['text_stopwords_stem2'] = base_tweets1['text_stopwords_stem2'].apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('rt')]))


###Contador de palabras - aca deberia eliminar las palabras que tienen un recuento menor a 5
df=pd.DataFrame(base_tweets1['text_stopwords_stem2'])
contador_palabras=df.text_stopwords_stem2.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
#Selecciona las que tienen mas de 5
contador_palabras=contador_palabras[(contador_palabras  <5)]

contador_palabras=contador_palabras.reset_index()


##genera una lista
palabras=contador_palabras['index'].tolist()
base_tweets1['text_final'] = base_tweets1['text_stopwords_stem2'].apply(lambda x: ' '.join([word for word in x.split() if word not in (palabras)]))


###ver ejemplos
i=711
print(base_tweets1['text'][i])
print(base_tweets1['text_punct'][i])
print(base_tweets1['text_stopwords'][i])
print(base_tweets1['text_stopwords_stem'][i])
print(base_tweets1['text_stopwords_stem2'][i])
print(base_tweets1['text_final'][i])


###Aplico



###Arma el tokenizer de twitter 
tw = TweetTokenizer()
##lo aplica a los tweets completos- no la uso - lo hace pero no me sirve para el vectorizer
tokens=base_tweets1['text_final'].apply(tw.tokenize)
se = pd.Series(tokens)
base_tweets1['tokenized'] = pd.DataFrame(se.values)

print(base_tweets1['text_final'][2])
print(base_tweets1['tokenized'][2])


###ARMO EL BAG OF WORDS
tf=base_tweets1['tokenized'].apply(pd.value_counts).fillna(0).astype(int)




###############3






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
model.fit(tf)



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
print("Model Perplexity: ", best_lda_model.perplexity(tf))



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




#############VER ESTO, ES UN PROBLEMITA
best_lda_model.n_topics=10



# Create Document - Topic Matrix
lda_output = best_lda_model.transform(tf)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_topics)]
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_topics)]



# index names
docnames = ["Doc" + str(i) for i in range(10000)]

# index names
docnames = ["Doc" + str(i) for i in range(10000)]


# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics

df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
df_topic_distribution



####VER####
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(best_lda_model, tf, tf, mds='tsne')
panel

pyLDAvis.save_html(panel,'../resultados/vis_sklearn.html')














# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_)
# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
# View
df_topic_keywords.head()





# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords












