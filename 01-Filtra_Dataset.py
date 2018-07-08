###Copiar Config_Local.py en el mismo directorio y cambiar por el path local abajo
import sys
sys.path.insert(0, '/Users/inesfrias/Documents/Posgrado/Textmining/LeyAborto/TPFinal/scripts')

from Config_Local import directorio_datos, archivo_tweets

###Librerias###
import pandas as pd


###Levanta datos
base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/data.tweet_tabulador.txt", sep='\t', encoding='latin1', low_memory=False)
###ELIMINO UN CASO QUE LO LEVANTO MAL
base_tweets1=base_tweets1.drop(base_tweets1.index[[169761]])
##RESETEO EL INDEX
base_tweets1=base_tweets1.reset_index(drop=True)



####PREPROCESAMIENTO####
###DIA HORA
base_tweets1['horario']=pd.to_datetime(base_tweets1['created_at'])
base_tweets1['horario'] = base_tweets1['horario'] - pd.Timedelta('03:00:00')
###elimino segundos
base_tweets1['horario'] = base_tweets1['horario'].apply(lambda t: t.replace(second=0))


###Elimina usuarios, paginas y rt - Lo cambio de orden porque estas referncias no permiten eliminar bien los duplicados
#Ejemplo: tweets iguales con distintos https
base_tweets1['text']=base_tweets1['text'].str.lower()  

base_tweets1['text_no_ref'] = base_tweets1['text']. \
    apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('@')]))
base_tweets1['text_no_ref'] = base_tweets1['text_no_ref'].\
    apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('http')]))
base_tweets1['text_no_ref'] = base_tweets1['text_no_ref'].\
    apply(lambda x: ' '.join([word for word in x.split() if not  word.startswith('rt')]))


###Saca caracteres especiales y hashtag
base_tweets1['text_punct'] = base_tweets1['text_no_ref'].apply(lambda x: ' '. \
            join([word for word in x.split() if not  word.startswith('#')]))
base_tweets1['text_punct'] = base_tweets1['text_punct'].str.replace(',',' ')
base_tweets1['text_punct'] = base_tweets1['text_punct'].str.replace('[^\w\s]','')

###Quitar los duplicados, mantener el ultimo##  Es min() o max()? que campo toma?
unique_tweets=base_tweets1.groupby(['text_punct']).min()
unique_tweets = unique_tweets[unique_tweets.text_punct != ""]
unique_tweets.dropna(subset=['text_punct'])



unique_tweets=unique_tweets.reset_index()
###convierte el df a csv

unique_tweets.columns
unique_tweets.drop(columns=['in_reply_to_user_id_str', 'lang',
                            'favorited', 'truncated',
       'id_str','in_reply_to_status_id_str','statuses_count',
       'followers_count', 'favourites_count', 'protected', 'time_zone','user_lang'])

unique_tweets.to_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/unique_tweets.txt",  header=True, sep='\t', encoding='latin1')



