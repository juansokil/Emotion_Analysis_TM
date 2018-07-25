# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 15:19:29 2018

@author: Juan
"""

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

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt


#####LEVANTA DATOS#####
base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/unique_tweets.txt", sep='\t', encoding='latin1', low_memory=False)
base_tweets1['horario']=pd.to_datetime(base_tweets1['horario'])
####delete nan#####
base_tweets1 = base_tweets1[~base_tweets1['text_punct'].isnull()]


###Cargo stopwords
from nltk.corpus import stopwords
stopwords=stopwords.words('spanish')
base_tweets1['text_punct'] = base_tweets1['text_punct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))


###Numeros
base_tweets1['text_punct']=base_tweets1['text_punct'].replace('\d+', 'NUM', regex=True)

###Aplica stemmer (No es muy lindo - habria que probar un lematizador)
#from nltk.stem import SnowballStemmer
#stemmer_spanish = SnowballStemmer('spanish')
#base_tweets1['text_punct'] = base_tweets1['text_punct'].apply(lambda x: ' '.join([stemmer_spanish.stem(word) for word in x.split()]))




####### TGOKENIZAR####
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(base_tweets1['text_punct']))

print(base_tweets1['text_punct'][:1])
print(data_words[:1])


#from gensim.corpora import Dictionary
#from gensim import corpora

#dictionary = corpora.Dictionary(data_words)
#print(len(dictionary))
#### vuela las palabras que aparecen muchos, o poco
#dictionary.filter_extremes()
#print(len(dictionary))



#####################LDA#######################
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=0.001,
                             max_df=0.70,
                             ngram_range=(1,2),
                             # minimum reqd occurences of a word 
                             #stop_words='english',             # remove stop words
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 2
                             )

data_vectorized = vectorizer.fit_transform(base_tweets1['text_punct'])
vectorizer_feature_names = vectorizer.get_feature_names()






###############LDA WITH SKLEARN
#####GRID PARAMETROS OPTIMOS#####

# Define Search Param
search_params = {'n_components': [2,3,4,5,6,7], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

######PARAMETROS############
####doc_topic_prior=alpha (mientras mas bajo menos topicos)
####learning decay = Parametro Kappa
####Topic word prior=Beta (mientras mas bajo menos palabras tiene cada topico)


GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=0.1,
             evaluate_every=-1, learning_decay=0.7, learning_method='online',
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1,
             n_topics=None, perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'n_topics': [2,3,4,5,6,7], 'learning_decay': [0.5, 0.7, 0.9]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=True)

# Do the Grid Search
model.fit(data_vectorized)
       
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

# Get Log Likelyhoods from Grid Search Output
n_topics = [2,3,4,5,6,7]
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


####Modelo optimo#####
lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=0.1,
             evaluate_every=-1, learning_decay=0.9,
             learning_method='online', learning_offset=10.0,
             max_doc_update_iter=100, max_iter=100, mean_change_tol=0.001,
             n_components=6, n_jobs=-1, n_topics=6, perp_tol=0.1,
             random_state=100, topic_word_prior=None,
             total_samples=1000000.0, verbose=True)






print(lda_model)  # Model attributes

lda_output = lda_model.fit_transform(data_vectorized)

# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))

# See model parameters
pprint(lda_model.get_params())






#############VER ESTO, ES UN PROBLEMITA
lda_model.n_topics=6

# column names
topicnames = ["Topic" + str(i) for i in range(lda_model.n_topics)]
topicnames = ["Topic" + str(i) for i in range(lda_model.n_topics)]



# index names
docnames = ["Doc" + str(i) for i in range(172072)]

# index names
docnames = ["Doc" + str(i) for i in range(172072)]


# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic


## Styling
#def color_green(val):
#    color = 'green' if val > .1 else 'black'
#    return 'color: {col}'.format(col=color)
#
#def make_bold(val):
#    weight = 700 if val > .1 else 400
#    return 'font-weight: {weight}'.format(weight=weight)


###convierte el df a df
df_document_topic.to_csv('base_topicos_twitter.csv',  header=True, sep='\t', encoding='latin1')

df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
df_topic_distribution


####VER####
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
panel
pyLDAvis.save_html(panel,'../resultados/vis_sklearn_twitter.html')


# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda_model.components_)
# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
# View
df_topic_keywords.head()



# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=50):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=50)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords


###convierte el df a df
df_topic_keywords.to_csv('palabras_topicos_twitter.csv',  header=True, sep='\t', encoding='latin1')

###asigno los valores de la serie a una nueva variable
se = pd.Series(df_document_topic['dominant_topic'])
base_tweets1['dominant_topic'] = se.values


base_tweets1.to_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/unique_tweets_lda.txt",  header=True, sep='\t', encoding='latin1')



