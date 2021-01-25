#Functions.py

#All the imports
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

#Some global variables
genre_list = ['Action','Adventure','Comedy','Crime','Drama','Fantasy','Horror','Romance','Thriller']
testSize = 0.3

#Classifiers introduction

#SVM
rating_modelSVM = svm.SVC(probability=True)
money_modelSVM = svm.SVC(probability=True)
#Decision Tree
rating_modelDTree = DecisionTreeClassifier(random_state=0)
money_modelDTree = DecisionTreeClassifier(random_state=0)
#K Nearest Neighbours
rating_modelKNN = KNeighborsClassifier(n_neighbors=5)
money_modelKNN = KNeighborsClassifier(n_neighbors=5)
#Adaboost
rating_modelAda = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
money_modelAda = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
#Naive Bayes
rating_modelNB = GaussianNB()
money_modelNB = GaussianNB()
#Multi Layer Perceptron
rating_modelMLP = MLPClassifier(solver="lbfgs",alpha=0.1,hidden_layer_sizes=(5,2),random_state=1)
money_modelMLP = MLPClassifier(solver="lbfgs",alpha=0.1,hidden_layer_sizes=(5,2),random_state=1)
#Random Forest
rating_modelRF = RandomForestClassifier(max_depth=2,random_state=0)
money_modelRF = RandomForestClassifier(max_depth=2,random_state=0)

#Dataset
dataset = pd.read_csv("copy8.csv",encoding="ISO-8859-1").dropna(axis=0)
#Feature Extraction
Dir = dataset['Director'].values
Act1 = dataset['Actor1'].values
Act2 = dataset['Actor2'].values
budget = dataset['Budget'].values
bclass = dataset['Bclass'].values
revenue = dataset['Revenue'].values
X1 = dataset['Action'].values
X2 = dataset['Adventure'].values
X3 = dataset['Comedy'].values
X4 = dataset['Crime'].values
X5 = dataset['Drama'].values
X6 = dataset['Fantasy'].values
X7 = dataset['Horror'].values
X8 = dataset['Romance'].values
X9 = dataset['Thriller'].values
Y = dataset['Result'].values
Y_dash = dataset['FRR'].values
final = dataset['Final'].values

#Getting the dataset values in format for model
X_rating_array = []
X_money_array = []

for x in range(0,len(Y)):
	list1 = []
	list1.append(Dir[x])
	list1.append(X1[x])
	list1.append(X2[x])
	list1.append(X3[x])
	list1.append(X4[x])
	list1.append(X5[x])
	list1.append(X6[x])
	list1.append(X7[x])
	list1.append(X8[x])
	list1.append(X9[x])
	X_rating_array.append(list1)
	
for x in range(0,len(Y)):
	list1 = []
	list1.append(Dir[x])
	list1.append(Act1[x])
	list1.append(Act2[x])
	list1.append(bclass[x])
	X_money_array.append(list1)

#Array maker
X_rating_array = np.array(X_rating_array)
Y_array = np.array(Y)

X_money_array = np.array(X_money_array)
Y_money_array = np.array(Y_dash)


#Training function
def train_classifiers():
	#Splitting the data set
	X_rtrain,X_rtest,Y_rtrain,Y_rtest=train_test_split(X_rating_array,Y_array,test_size=testSize,random_state=0)
	X_mtrain,X_mtest,Y_mtrain,Y_mtest=train_test_split(X_money_array,Y_money_array,test_size=testSize,random_state=0)
	
	#Fitting the model	
	rating_modelSVM.fit(X_rtrain,Y_rtrain)
	money_modelSVM.fit(X_mtrain,Y_mtrain)
	
	rating_modelKNN.fit(X_rtrain,Y_rtrain)
	money_modelKNN.fit(X_mtrain,Y_mtrain)
	
	rating_modelDTree.fit(X_rtrain,Y_rtrain)
	money_modelDTree.fit(X_mtrain,Y_mtrain)
	
	rating_modelAda.fit(X_rtrain,Y_rtrain)
	money_modelAda.fit(X_mtrain,Y_mtrain)
	
	rating_modelNB.fit(X_rtrain,Y_rtrain)
	money_modelNB.fit(X_mtrain,Y_rtrain)
	
	rating_modelMLP.fit(X_rtrain,Y_rtrain)
	money_modelMLP.fit(X_mtrain,Y_mtrain)
	
	rating_modelRF.fit(X_rtrain,Y_rtrain)
	money_modelRF.fit(X_mtrain,Y_mtrain)
	
#Function returning binary genre list
def genre_classifier(genres):	#List containing Genres
	genres.sort()
	binary_genre = [0,0,0,0,0,0,0,0,0]
	for j in range(0,len(genre_list)):
		for g in genres:
			if genre_list[j] == g and binary_genre[j] == 0:
				binary_genre[j] = 1
	return(binary_genre)	#Returns the binarized version

#Function to classify budget into appropriate class
def budget_classifier(budget_movie):
	if budget_movie < 170000000:
		bclass_b = 1
	elif budget_movie >= 170000000 and budget_movie < 340000000:
		bclass_b = 2
	elif budget_movie >=340000000 and budget_movie < 510000000:
		bclass_b = 3
	elif budget_movie >=510000000:
		bclass_b = 4
	return(bclass_b)

#Function to convert list item into string	
def tostr(a):
	st1 = ""
	for s in a:
		st1 += s.replace('\n','') + ' '
	return st1
	
def director_actor_dict_maker():
	f1 = open("director-dict.txt","r")
	f2 = open("actors-dict.txt","r")

	director = {}
	actor = {}
	
	dire = f1.readlines()
	acto = f2.readlines()
	
	for d in dire:
		di = d.split(' ')
		n = int(di[0])
		k = tostr(di[1:]).strip()
		director.update({n:k})
		
	for d in acto:
		di = d.split(' ')
		n = int(di[0])
		k = tostr(di[1:]).strip() 
		actor.update({n:k})
		
	
	f1.close()
	f2.close()
	
	return(director,actor)

#Function to give directors and actors their numbers
def director_actor_numbers(dii,aa1,aa2):
	a1n,a2n,dn = -1,-1,-1
	#Get the dictionary
	director,actor = director_actor_dict_maker()
	
	for n,d in director.items():
		if dii == d:
			dn = n
			break

	for n,a1 in actor.items():
		if aa1 == a1:
			a1n = n
			break
	
	for n,a1 in actor.items():
		if aa2 == a1:
			a2n = n
			break
	
	return(dn,a1n,a2n)
	
#Function to get predictions
def jhol_predict(X_rating_test, X_money_test):
	
	class_list = []
	
	ratestSVM = rating_modelSVM.predict(X_rating_test)
	motestSVM = money_modelSVM.predict(X_money_test)
	if ratestSVM.item(0) == 1 and motestSVM.item(0) == 1:
		final = 3
	if ratestSVM.item(0) == 1 and motestSVM.item(0) == 0:
		final = 2
	if ratestSVM.item(0) == 0 and motestSVM.item(0) == 1:
		final = 1
	if ratestSVM.item(0) == 0 and motestSVM.item(0) == 0:
		final = 0
	class_list.append(final)
		
	ratestKNN = rating_modelKNN.predict(X_rating_test)
	motestKNN = money_modelKNN.predict(X_money_test)
	if ratestKNN.item(0) == 1 and motestKNN.item(0) == 1:
		final = 3
	if ratestKNN.item(0) == 1 and motestKNN.item(0) == 0:
		final = 2
	if ratestKNN.item(0) == 0 and motestKNN.item(0) == 1:
		final = 1
	if ratestKNN.item(0) == 0 and motestKNN.item(0) == 0:
		final = 0
	class_list.append(final)
	
	ratestDTree = rating_modelDTree.predict(X_rating_test)
	motestDTree = money_modelDTree.predict(X_money_test)
	if ratestDTree.item(0) == 1 and motestDTree.item(0) == 1:
		final = 3
	if ratestDTree.item(0) == 1 and motestDTree.item(0) == 0:
		final = 2
	if ratestDTree.item(0) == 0 and motestDTree.item(0) == 1:
		final = 1
	if ratestDTree.item(0) == 0 and motestDTree.item(0) == 0:
		final = 0
	class_list.append(final)
	
	ratestAda = rating_modelAda.predict(X_rating_test)
	motestAda = money_modelAda.predict(X_money_test)
	if ratestAda.item(0) == 1 and motestAda.item(0) == 1:
		final = 3
	if ratestAda.item(0) == 1 and motestAda.item(0) == 0:
		final = 2
	if ratestAda.item(0) == 0 and motestAda.item(0) == 1:
		final = 1
	if ratestAda.item(0) == 0 and motestAda.item(0) == 0:
		final = 0
	class_list.append(final)

	ratestNB = rating_modelNB.predict(X_rating_test)
	motestNB = money_modelNB.predict(X_money_test)
	if ratestNB.item(0) == 1 and motestNB.item(0) == 1:
		final = 3
	if ratestNB.item(0) == 1 and motestNB.item(0) == 0:
		final = 2
	if ratestNB.item(0) == 0 and motestNB.item(0) == 1:
		final = 1
	if ratestNB.item(0) == 0 and motestNB.item(0) == 0:
		final = 0
	class_list.append(final)
	
	ratestMLP = rating_modelMLP.predict(X_rating_test)
	motestMLP = money_modelMLP.predict(X_money_test)
	if ratestMLP.item(0) == 1 and motestMLP.item(0) == 1:
			final = 3
	if ratestMLP.item(0) == 1 and motestMLP.item(0) == 0:
			final = 2
	if ratestMLP.item(0) == 0 and motestMLP.item(0) == 1:
			final = 1
	if ratestMLP.item(0) == 0 and motestMLP.item(0) == 0:
			final = 0
	class_list.append(final)

	ratestRF = rating_modelRF.predict(X_rating_test)
	motestRF = money_modelRF.predict(X_money_test)
	if ratestRF.item(0) == 1 and motestRF.item(0) == 1:
			final = 3
	if ratestRF.item(0) == 1 and motestRF.item(0) == 0:
			final = 2
	if ratestRF.item(0) == 0 and motestRF.item(0) == 1:
			final = 1
	if ratestRF.item(0) == 0 and motestRF.item(0) == 0:
			final = 0
	class_list.append(final)
	
	#Jhol begins here
	class_list.sort()
	return(class_list[-1])

#Function to compare values of final class
def final_result(final):
	if final == 3:
		return("Would be a box office hit and be well received by critics.")
	elif final == 2:
		return("Would be well received by critics, might not do well at the box office.")
	elif final == 1:
		return("Might not be well received by critics, but may be a box office hit.")
	elif final == 0:
		return("Low chances of good run at box office and might not be well received by critics.")