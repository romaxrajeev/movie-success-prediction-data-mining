#This file will be used for Flask Integration

from Function import *
from Twitter import *
from Youtube import *

#Train the classifiers
train_classifiers()

#Get user input
u_id = "2F6CGSGEWV" #Need to be changed later

#Open file and check if the laste condition is fulfilled
f = open("D:\\MiniProject\\Final\\YT-Records.txt","r")
all_info = f.readlines()
for info in all_info:
	entry = info.replace('\n','')
	entry_list = entry.split(',')
	t_id = entry_list[0]
	if t_id == u_id:
		req = entry_list
		last = int(entry_list[2])
		break

if last == 38:
	#Then only print other info
	movie = input("Enter Movie Name:")
	diirector = input("Director:")
	aactor1 = input("Actor1 (Male):")
	aactor2 = input("Actor2 (Female):")
	budget_movie = int(input("Enter Budget:"))
	genres = input("Enter genres separated by comma and no spaces:").split(',')
	youtube_trailer = input("Enter Youtube Trailer Link:")
	
	#Using functions
	bclass_b = budget_classifier(budget_movie)
	binary_genre = genre_classifier(genres)
	dn,a1n,a2n = director_actor_numbers(diirector,aactor1,aactor2)
	
	#Making user input array
	list1 = [dn,binary_genre[0],binary_genre[1],binary_genre[2],binary_genre[3],binary_genre[4],binary_genre[5],binary_genre[6],binary_genre[7],binary_genre[8]]
	list2 = [dn,a1n,a2n,bclass_b]
	X_rating_test = np.array(list1).reshape(1,-1)
	X_money_test = np.array(list2).reshape(1,-1)
	final = jhol_predict(X_rating_test, X_money_test)
	
	#Twitter Analyzer at work
	twitter_score = twitter_analyzer(movie)
	
	#Youtube Analyzer
	yt_views,yt_likes,yt_dislikes = youtube_analyzer(req)	#Giving record string input
	
	#Final output
	print("Here are the results:")
	print("Prediction Model says:",final_result(final))
	print("Twitter Analyzer says:",twitter_verdict(twitter_score))
	print("Youtube Analyzer says:",yt_verdict(yt_views,yt_likes,yt_dislikes))
	
else:
	print("Still processing....")