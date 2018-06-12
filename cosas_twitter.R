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




setwd("~/Emotion_Analysis_TM")
source('./my_oauth.r', encoding = 'latin1')

#tw = twitteR::searchTwitter('#VotenAbortoLegal + #SalvemosLas2Vidas', n = 20, since = '2018-06-12 00:22:51', until='2016-06-12 02:22:51', retryOnRateLimit = 1e3)
#tweets.df <- twListToDF(tw)

filterStream(file.name="./tweets/tweets_abortolegal.json", locations = c(-72,-55,-55,-22), language = "es", track = "AbortoLegal", timeout = 100,  oauth = my_oauth)
filterStream(file.name="./tweets/tweets_2vidas.json", locations = c(-72,-55,-55,-22),language = "es",  track = "SalvemosLas2Vidas", timeout = 100,  oauth = my_oauth)

tweetslegal.df <- parseTweets("./tweets/tweets_abortolegal.json", verbose = FALSE)
tweets2vidas.df <- parseTweets("./tweets/tweets_2vidas.json", verbose = FALSE)


map.data <- map_data("world")
points <- data.frame(x = as.numeric(tweetslegal.df$place_lon), y = as.numeric(tweetslegal.df$place_lat))
points2 <- data.frame(x = as.numeric(tweets2vidas.df$place_lon), y = as.numeric(tweets2vidas.df$place_lat))
ggplot(map.data) + geom_map(aes(map_id = region), map = map.data, fill = "white", 
                            color = "grey20", size = 0.25) + expand_limits(x = map.data$long, y = map.data$lat) + 
  theme(axis.line = element_blank(), axis.text = element_blank(), axis.ticks = element_blank(), 
        axis.title = element_blank(), panel.background = element_blank(), panel.border = element_blank(), 
        panel.grid.major = element_blank(), plot.background = element_blank(), 
        plot.margin = unit(0 * c(-1.5, -1.5, -1.5, -1.5), "lines")) + 
  geom_point(data = points, aes(x = x, y = y), size = 2, alpha = 1, color = "green") +
geom_point(data = points2, aes(x = x, y = y), size = 2, alpha = 1, color = "blue")




