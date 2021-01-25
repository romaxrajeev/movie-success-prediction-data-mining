#Twiiter

#Importing Twitter and NLP Libraries 
import tweepy
from textblob import TextBlob

def twitter_analyzer(movie_name):
	#Authorization keys declaration
	consumer_api_key = ""
	consumer_api_secret = ""
	access_token = ""
	access_token_secret = ""
	
	#Getting Access to Twitter
	auth = tweepy.OAuthHandler(consumer_api_key, consumer_api_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)
	
	#To ensure that only movie related tweets are received
	tweets = api.search(movie_name+" Movie")
	overall_score = 0
	
	for tweet in tweets:
		analysis = TextBlob(tweet.text)
		#Getting polarity of each tweet
		polarity = analysis.sentiment.polarity
		if polarity > 0:
			overall_score += 1
		elif polarity == 0:
			overall_score = overall_score
		elif polarity < 0:
			overall_score -= 1
		
	return(overall_score)

def twitter_verdict(score):
	if score < 0:
		return("More of negative social hype for the movie")
	elif score == 0:
		return("Neutral hype for the movie")
	elif score > 0:
		return("Good positive social media hype for the movie")
