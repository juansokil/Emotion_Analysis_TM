#install.packages("twitteR")
#install.packages("streamR")
#install.packages("maps")

library(twitteR)
library(streamR)
library(ROAuth)
library(maps)
library(ggplot2)
library(grid)
library(maps)
library(data.table)
library(tm)


setwd("~/Emotion_Analysis_TM")
source('./my_oauth.r', encoding = 'latin1')


#####Download Tweets####
i=1
while(TRUE)
{
  i=i+1
  filterStream( file=paste0("./tweets/tweets_aborto",i,".json"),
                track = c("#AbortoLegalYa","#AbortoLegalSeguroYGratuito","#13JAbortoLegal","#VotenAbortoLegal","#SalvemosLas2Vidas","#ArgentinaEsProVida","#SiALaVida","#NoAlAborto","#AbortoSesionHistorica"), locations = c(-72,-55,-55,-22), language = "es", timeout=600, oauth=my_oauth)
}



#####Merge files####
data.tweet=NULL
i=1
while(TRUE)
{
  i=i+1
  print(i)
  print(paste0("tweets/tweets_aborto",i,".json"))
  if (is.null(data.tweet))
    data.tweet=data.table(parseTweets(paste0("tweets/tweets_aborto",i,".json")))
  else
    data.tweet=rbind(data.tweet,data.table(parseTweets(paste0("tweets/tweets_aborto",i,".json"))))
}



str(data.tweet)

###Elimina duplicados
data.tweet <- unique(data.tweet)



####calcula maximo RT y minimo RT####
data.tweet[,min_RT:=min(retweet_count),by=text]
data.tweet[,max_RT:=max(retweet_count),by=text]


###Formato Texto
data.tweet$text=iconv(data.tweet$text,from="UTF-8",to="ASCII//TRANSLIT")

##Remove double whitespaces
data.tweet$text=stripWhitespace(data.tweet$text)

###Muestra los 10 primeros tweets y los primeros timestamp###
head(data.tweet$text,10)
head(data.tweet$created_at,10)

###veo las variables
str(data.tweet)



###Guardo solo los georeferenciados###
data.tweet_georef<-subset(data.tweet, (!is.na(data.tweet$place_lat)))




#dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")
#write.table(data.tweet_georef, file= data.tweet_georef, row.names=FALSE, col.names=TRUE, quote=FALSE, sep="\t", eol = "\r\n")

write.csv(data.tweet_georef, 'salida.csv')
#leo el dataset

  


#rm(list = c('leyaborto.df','points','pais7','pais8','pais9'))



##http://enhancedatascience.com/2017/07/17/twitter-analysis-using-r/










map.data <- map_data("world")
points <- data.frame(x = as.numeric(data.tweet_georef$place_lon), y = as.numeric(data.tweet_georef$place_lat))
#points2 <- data.frame(x = as.numeric(tweets2vidas.df$place_lon), y = as.numeric(tweets2vidas.df$place_lat))
ggplot(map.data) + geom_map(aes(map_id = region), map = map.data, fill = "white", 
                            color = "grey20", size = 0.25) + expand_limits(x = map.data$long, y = map.data$lat) + 
  theme(axis.line = element_blank(), axis.text = element_blank(), axis.ticks = element_blank(), 
        axis.title = element_blank(), panel.background = element_blank(), panel.border = element_blank(), 
        panel.grid.major = element_blank(), plot.background = element_blank(), 
        plot.margin = unit(0 * c(-1.5, -1.5, -1.5, -1.5), "lines")) + 
  geom_point(data = points, aes(x = x, y = y), size = 3, alpha = 0.6, color = "dark green")
#+geom_point(data = points2, aes(x = x, y = y), size = 2, alpha = 0.6, color = "light blue")




