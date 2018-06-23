
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt



###Levanta datos
base_tweets1 = pd.read_csv("C:/Users/Juan/Documents/GitHub/Emotion_Analysis_TM/dataset/data.tweet_tabulador.txt", sep='\t', encoding='latin1', low_memory=False)
base_tweets1.columns
####es la unica forma que encontré de hacerlo, es malisima.... se hacen rangos de 10 minutos sobre la base
###mantiene el dia, hora, minutos (solo el primer digito)
base_tweets1['horario']=base_tweets1['created_at'].str.slice(8,15)


##POR AHORA QUEDA PENDIENTE
#localidades_reg_ex = base_tweets1_reduc.groupby('location').agg('size').sort_values(ascending=False)
#nombres_reg_ex = base_tweets1_reduc.groupby('name').agg('size').sort_values(ascending=False)

                      

#####Tweets por minutos###TOTAL - POSITIVOS - NEGATIVOS #
base_tweets1.groupby('horario').agg('size').plot(color='black')
base_tweets1[base_tweets1['#AbortoLegalYa'] | base_tweets1['#AbortoLegalSeguroYGratuito'] | base_tweets1['#AbortoSeraLey'] | base_tweets1['#QueSeaLey'] | base_tweets1['#13JAbortoLegal'] | base_tweets1['#VotenAbortoLegal']  ].groupby('horario').agg('size').plot(color='green')
base_tweets1[base_tweets1['#SalvemosLas2Vidas'] | base_tweets1['#ArgentinaEsProVida']  | base_tweets1['#NoAlAbortoEnArgentina'] | base_tweets1['#SiALaVida'] | base_tweets1['#NoAlAbortoEnArgentina'] ].groupby('horario').agg('size').plot(color='blue')                 
####aca le inclui algunos diputados
base_tweets1[base_tweets1['text'].str.contains('Massot', 'massot', regex=False)  ].groupby('horario').agg('size').plot(color='grey')
base_tweets1[base_tweets1['text'].str.contains('Carrio', 'carrio', regex=False)  ].groupby('horario').agg('size').plot(color='yellow')
base_tweets1[base_tweets1['text'].str.contains('olmedo', 'Olmedo' , regex=False)  ].groupby('horario').agg('size').plot(color='purple')
base_tweets1[base_tweets1['text'].str.contains('belledone', 'Belledone' , regex=False)  ].groupby('horario').agg('size').plot(color='orange')
plt.show()






#AbortoSeraLey


### tiene el problema de que el horario argentino es -3, o sea arranca a las 14:30 segun horario internacional, que en realidad es 11:30 nuestro
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
plt.show()

