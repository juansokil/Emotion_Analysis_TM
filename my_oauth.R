

api_key <- "rYct9vETBZJ9aRFiBMrfqHfRD"
api_secret <- "c2sjBrxR6sJpOx9vXkNj3gaxI7O9lbWi4rx8opdD90zVuVZCRM"
access_token <- "3022468097-QWi0iyB2yg4HcbffGpjKKrSj1EOuN5rfcrMaie0"
access_token_secret <- "UU1ArPSZwg2gdEEIcTfUeOBrhxJKjTOKHSRHoAZGWTp7v"

requestURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "https://api.twitter.com/oauth/authorize"
consumerKey <- "rYct9vETBZJ9aRFiBMrfqHfRD"
consumerSecret <- "c2sjBrxR6sJpOx9vXkNj3gaxI7O9lbWi4rx8opdD90zVuVZCRM"



setup_twitter_oauth(api_key, api_secret, access_token, access_token_secret)



my_oauth <- OAuthFactory$new(consumerKey=consumerKey,
                             consumerSecret=consumerSecret, requestURL=requestURL,
                             accessURL=accessURL, authURL=authURL)

my_oauth$handshake(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl"))



