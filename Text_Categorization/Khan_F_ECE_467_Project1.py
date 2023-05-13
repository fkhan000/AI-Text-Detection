#Fakharyar Khan
#March 20th, 2023
#Natural Language Processing
#Professor Sable


						  ##### Project 1: Text Classification #####


#import the necessary libraries                        
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import wordnet 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
stopwords = stopwords.words('english')
punctuation = list(punctuation)



#prompt the user for the filenames for the list of training and testing documents
trainDoc = input("Please enter the filename for the list of training documents: ")
testDoc =  input("Please enter the filename for the list of testing documents: ")

#read these filenames and their corresponding categories into a dataframe
train = pd.read_csv(trainDoc, sep=" ", header=None, names = ["Filename", "Category"])

#for the test dataframe the categories column will be blank and we'll fill it in after we trained our classifier
test = pd.read_csv(testDoc, sep=" ", header=None, names = ["Filename", "Category"])


lemmatizer = WordNetLemmatizer()
tweet_tokenizer = TweetTokenizer()
stemmer = PorterStemmer()


def getStatistics(trainSet):
  #here we're going to calculate the probability of a document being in a certain
  #category by first looking at the fraction of documents in the training set that
  #are in that category
	probCateg = trainSet["Category"].value_counts()

	probCateg = probCateg/probCateg.sum()

  #we create a dictionary that maps a category to the probability of a document being
  #in that category

	probCateg = probCateg.to_dict()


  #to calculate p(t|c) we need to find the probability of a term occuring in 
  #a category

  #So to quickly get the probability of a word appearing in a document of a certain category
  #I used a dictionary which maps the word to another dictionary which in turn maps the category
  #to "C(t|c)". This might not be as efficient as having the dictionary map to a list of 
  #prob values (lookup is still O(1) but hashing the key is probably more computationally expensive than retrieving 
  #element index) but in my opinion it makes the data structure work very well with pandas 
  #so that training my classifier only takes a few lines of code.
	wordFreq = {}


  

  #for each data point in our training set
	for index, row in trainSet.iterrows():

	  #we read in the document
		with open(row["Filename"], 'r') as f:
			text = f.read()

			f.close()

	 	#and split the document into tokens
	 	#then we're going to reduce all of the words in the document
	 	#to their base form and then make all of the letters lower case. 
	 	#This helps make our assumption that the probabilities of each individual
	 	#word appearing in the document are independent of one another more valid
	 	#as if you see the word robbery in a document then you're also likely to see
	 	#words like robbed, robbing, and rob in the document. Additionally, this will
	 	#make training and classification of our model much faster as there are significantly 
		#less words that it has to calculate probabilities for

		#For the third corpus I found that using stemming gave me a higher accuracy than if I used lemmatization.
		#Additionally I found that counting the number of times the word appears in a document
	 	#instead of just if it occurs in the document signifcantly increases the accuracy by around 4% for corpus 3
	 	#And I was playing around with tokenizers and found that the tweet tokenizer, a tokenizer used to tokenize tweets,
	 	#worked very well with corpus 3 although I was planning on using it for corpus 2

		if "Ent" not in probCateg.keys():
			tokens = nltk.word_tokenize(text)
			tokens = list(set(tokens))
		else:
			tokens = tweet_tokenizer.tokenize(text)


		tokens = [token.lower() for token in tokens]
		#I also removed stop words and punctuation from the documents since they only seemed
	 	#to hurt the model by adding noise and it will also make training and classification
	 	#faster as well.
	 	
	  	#Additionallly I found that strangely enough using tokens that were only words in corpus 3 increased its performance
	  	#and for corpus 1, the best performance was found when neither lemmatization nor stemming was used
		if "Ent" in probCateg.keys():
			tokens = [token for token in tokens if  (token not in stopwords) and token.isalpha()]
			tokens = [stemmer.stem(word) for word in tokens]

		else:
			tokens = [token for token in tokens if (token not in punctuation) and (token not in stopwords)]

			if "O" in probCateg.keys():
				tokens = [(lemmatizer.lemmatize(word)) for word in tokens]
			tokens = list(set(tokens))


	  

		
	  #for each word
		for word in tokens:
		#if the word isn't yet in the dictionary 
			if not (word in wordFreq):
		
		  		#initialize an entry in the dictionary that maps that word
		  		#to a dictionary that maps the names of the categories to 0s
		  		#since so far none of the categories have had that word
				wordFreq[word] = dict(zip(probCateg.keys(), [0]*len(probCateg.keys())))
		
		
			#then if the word is in that document. We increment the # of occurences of that term
			#in this category by 1
			wordFreq[word][row["Category"]] += 1

	  
	return wordFreq, probCateg
#in this function, we classify the document
#the function takes in the name of the file in which the document is in
#and a tunable paramter that's used to account for words that don't
#appear in the training set but are present in the testing set
#The function returns the category that the document is most likely
#to be
def classifyDoc(wordFreq, probCateg, filename, eps, trainSize):
	#open and read in the file
	with open(filename, 'r') as f:
		text = f.read()

		f.close()
  
	#and we do the same text processing that we did for the documents in the training set

	if "Ent" not in probCateg.keys():
		tokens = nltk.word_tokenize(text)
		tokens = list(set(tokens))
	else:
		tokens = tweet_tokenizer.tokenize(text)

	tokens = [token.lower() for token in tokens]

	if "Ent" in probCateg.keys():
		tokens = [token for token in tokens if  (token not in stopwords) and token.isalpha()]
		tokens = [stemmer.stem(word) for word in tokens]

	else:
		tokens = [token.lower() for token in tokens]
		tokens = [token for token in tokens if (token not in punctuation) and (token not in stopwords)]

		if "O" in probCateg.keys():
			tokens = [(lemmatizer.lemmatize(word)) for word in tokens]
		tokens = list(set(tokens))




	#since we're going to multiply several very small numbers together, we run 
	#the risk of the value rounding to 0. To avoid this, we take the log probability
	#instead

	#we're going to calculate the log probability for each category
	#so we can hold these values in a dictionary that maps the categories to 
	#their probabilities for that word

	logProb = dict(zip(probCateg.keys(), [math.log(probCateg[key]) for key in probCateg.keys()]))
	
	#check to see whether or not we're working with the third corpus
	ThirdCorp = "Ent" in probCateg.keys()
  #for each category
	for categ in probCateg.keys():
	#to get p(t|c) we divide by total number of documents that are of that category

		denom = trainSize*probCateg[categ]

		#if we're on the third corpus
		if ThirdCorp:
			#we apply the full form of laplace smoothing
			denom += eps
		#for each word in the document
		for word in tokens:
	  	#if we encountered the word before
			if word in wordFreq:
				#we get C(t|c) from the dictionary and add the log of it to the corresponding
				#log probability plus the smoothing parameter.
				logProb[categ] += math.log(wordFreq[word][categ] + eps)
		
			  
	  		#However, if the word isn't in that category, that would mean that p(t|c) = 0
	  		#which would mean that p(c|d) = 0 which is unreasonable. Instead,
	  		#we add a small constant, eps, to all of the term frequencies for all of the categories
			else:
				logProb[categ] += math.log(eps)

	  		#and then here we divide by denom to get p(t|c) which for log prob is the same as subtracting
			logProb[categ] -= math.log(denom)
		

  #after we have calculated the log probabilities we return the category with 
  #the highest probability
	return max(logProb, key=logProb.get) 

#this function takes in a list of documents and the smoothing parameter eps
#and returns a list of classifications made
def predict(wordFreq, probCateg, predictors, eps, trainSize):
  
	categ = []

  	#for each documentf
	for name in predictors:

	#we classify it and add it to the list of classifications
		categ.append(classifyDoc(wordFreq, probCateg, name, eps, trainSize))
  
  	#and then return that list
	return categ



#here we tune the smoothing parameter using 4-fold cross validation
#the function returns the optimal value for the paramter found
#After experimenting with values, I found that the optimal value
#was usually very small and less than 0.05. 

def tuneModel(folds):

  
	bestEps = 0.01
	minError = 1
  
  	#we split the training set into 4 pieces

  	#we then get the wordFreq dictionaries from the getStatistics function for each training set
	wordStats = []

	for index in range(folds):
		trainSet = pd.concat([train.iloc[math.floor(idx*len(train.index)/folds): math.ceil((idx + 1)*len(train.index)/folds)] for idx in range(folds) if idx != index])

		wordStats.append(getStatistics(trainSet))

	alpha = 1

	#for each value of epsilon
	for eps in np.arange(0.01*alpha, 0.1*alpha, 0.005*alpha):
	
		#initialize the error to 0
		error = 0

		#for each piece of the training set
		for index in range(folds):
	  		#we're going to make one of them the validation set and 
	  		#merge the other three into our training set
			validation = train.iloc[math.floor(index*len(train.index)/folds): math.ceil((index + 1)*len(train.index)/folds)]

	  
	 		#get a list of the ground truth labels for the validation set
			actual = validation["Category"].to_list()

			#we train on the 3 other pieces and get the predictions made for the validation set
			#using that value of eps. 
			predictions = predict(wordStats[index][0], wordStats[index][1], validation["Filename"].to_list(), eps, int(len(train.index)*(folds-1)/folds))

	  		#and then look at what fraction of the documents the classifier predicted
	  		#incorrectly
	  
			error += sum([0 if actual[idx] == predictions[idx] else 1 for idx in range(len(actual))])/len(actual)

		#once we have gone through every piece, we divide the total error by 4 to 
		#get the average error
		error /= folds

		#and if the error is less than the previous minimum error found
		if error < minError:
	  		#we make bestEps that value of epsilon
	  		bestEps = eps
	  		minError = error

	return bestEps


#this is the main driver function that predicts the classes for all of the 
#documents in the testing set and then prompts the user for a file
#that it can dump those classifications into
def driver():
  
  #get the optimal value for the smoothing parameter
  eps = tuneModel(8)
  wordStat = getStatistics(train)

  #get the labels
  labels = predict(wordStat[0], wordStat[1], test["Filename"], eps, len(train.index))


  #fill in the empty column in the test dataframe
  test["Category"] = labels

  #prompt user for filename
  outputFile = input("Please enter the filename that you would like the labels for the test documents to be placed in: ")

  test.to_csv(outputFile, header = None, index = None, sep = ' ', mode = 'w')
  
driver()
