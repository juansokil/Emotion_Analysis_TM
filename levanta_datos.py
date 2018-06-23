
import pandas as pd
import numpy as np
import re

###Levanta datos
base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/data.tweet_tabulador.txt", sep='\t', encoding='latin1', low_memory=False)
base_tweets1.columns

####base reducida, para explorarla mejor
base_tweets1_reduc=base_tweets1.head(n=20000)


####identificar ubicacion y genero###
localidades_reg_ex = base_tweets1_reduc.groupby('location').agg('size').sort_values(ascending=False)
nombres_reg_ex = base_tweets1_reduc.groupby('name').agg('size').sort_values(ascending=False)




emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)

