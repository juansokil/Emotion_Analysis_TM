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


setwd("~/GitHub/Emotion_Analysis_TM")
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


###Elimina duplicados
data.tweet <- unique(data.tweet)


####calcula maximo RT y minimo RT####
data.tweet[,min_RT:=min(retweet_count),by=text]
data.tweet[,max_RT:=max(retweet_count),by=text]

###Formato Texto
data.tweet$text=iconv(data.tweet$text,from="UTF-8",to="ASCII//TRANSLIT")
data.tweet$text=iconv(data.tweet$name,from="UTF-8",to="ASCII//TRANSLIT")
data.tweet$text=iconv(data.tweet$location,from="UTF-8",to="ASCII//TRANSLIT")


##Remove double whitespaces
data.tweet$text=stripWhitespace(data.tweet$text)
data.tweet$text=stripWhitespace(data.tweet$name)
data.tweet$text=stripWhitespace(data.tweet$location)


###Muestra los 10 primeros tweets y los primeros timestamp###
head(data.tweet$text,10)
head(data.tweet$created_at,10)

####Borra variables
data.tweet$expanded_url <- NULL
data.tweet$url <- NULL
data.tweet$user_url <- NULL
data.tweet$description  <- NULL
data.tweet$source  <- NULL

###veo las variables
str(data.tweet)





####Limpieza####

words <- c("#AbortoLegalYa","#AbortoLegalSeguroYGratuito","#13JAbortoLegal","#VotenAbortoLegal",
           "#SalvemosLas2Vidas","#ArgentinaEsProVida","#SiALaVida","#NoAlAborto","#AbortoSesionHistorica",
           "#QueSeaLey","#AbortoSeraLey","#NoAlAbortoEnArgentina")


tabla <- as.data.frame(sapply(words, grepl, data.tweet$text))

###Junta###
data.tweet <- cbind(data.tweet, tabla)


data.tweet2 = data.tweet[data.tweet$`#AbortoLegalYa` == TRUE |
                        data.tweet$`#AbortoLegalSeguroYGratuito` == TRUE |
                        data.tweet$`#13JAbortoLegal` == TRUE |
                        data.tweet$`#VotenAbortoLegal` == TRUE |
                        data.tweet$`#SalvemosLas2Vidas` == TRUE |
                        data.tweet$`#ArgentinaEsProVida` == TRUE |
                        data.tweet$`#SiALaVida` == TRUE |
                        data.tweet$`#NoAlAborto` == TRUE |
                        data.tweet$`#AbortoSesionHistorica` == TRUE |
                          data.tweet$`#QueSeaLey` == TRUE |
                          data.tweet$`#AbortoSeraLey` == TRUE |
                          data.tweet$`#NoAlAbortoEnArgentina` == TRUE ]




###selecciono solo los georeferenciados###
data.tweet_georef<-subset(data.tweet2, (!is.na(data.tweet2$place_lat)))


###Guarda las bases de tweets
write.table(data.tweet2, file="./dataset/data.tweet.txt", row.names=FALSE, col.names=TRUE, quote=TRUE, sep="\t", eol = "\r\n")
write.table(data.tweet_georef, file="./dataset/data.tweet_georef.txt", row.names=FALSE, col.names=TRUE, quote=TRUE, sep="\t", eol = "\r\n")


###MAPA####

map.data <- map_data("world")
points <- data.frame(x = as.numeric(data.tweet_georef$place_lon), y = as.numeric(data.tweet_georef$place_lat))
ggplot(map.data) + geom_map(aes(map_id = region), map = map.data, fill = "white", 
                            color = "grey20", size = 0.25) + expand_limits(x = map.data$long, y = map.data$lat) + 
  theme(axis.line = element_blank(), axis.text = element_blank(), axis.ticks = element_blank(), 
        axis.title = element_blank(), panel.background = element_blank(), panel.border = element_blank(), 
        panel.grid.major = element_blank(), plot.background = element_blank(), 
        plot.margin = unit(0 * c(-1.5, -1.5, -1.5, -1.5), "lines")) + 
  geom_point(data = points, aes(x = x, y = y), size = 3, alpha = 0.6, color = "dark green")










##http://enhancedatascience.com/2017/07/17/twitter-analysis-using-r/



