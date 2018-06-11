

#install.packages("twitteR")
library(twitteR)
#install.packages("streamR")
library(streamR)
library(ROAuth)


setwd("~/Emotion_Analysis")
source('./my_oauth.r', encoding = 'latin1')



## capture 10 tweets mentioning the "Rstats" hashtag
filterStream( file.name="tweets_rstats.json",
              track="aborto", tweets=10, oauth=my_oauth )

#my_oauth <- list(consumer_key=api_key, consumer_secret=api_secret, access_token, access_token_secret)


## capture 10 tweets mentioning the "Rstats" hashtag
filterStream( file.name="tweets_rstats.json",
              track="aborto", tweets=10,  language="es", oauth=my_oauth )




#tweets <- searchTwitter("aborto", n=100, language="es")
#tweets.df <- twListToDF(tweets)




