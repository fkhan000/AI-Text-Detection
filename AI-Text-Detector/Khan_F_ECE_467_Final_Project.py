#Fakharyar Khan
#May 2nd, 2023
#Natural Language Processing
#Professor Sable

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from string import punctuation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
stopwords = stopwords.words('english')
punctuation = list(punctuation)


#read in the dataset into a dataframe
df = pd.read_csv("GPT-wiki-intro.csv")
#we're going to reformat the dataset and apply text preprocessing
#and normalization

#only take rows in which the AI generated text has at least 150 words
#I found that the performance of the classifier decreases significantly
#when I kept small text samples in the dataset. The ideal and most common 
#application for this classifier in my opinion for determining if something 
#like ChatGPT was used to write a student's essays and in those scenarios
#you can expect to have more than 150 words


df = df[df["generated_intro_len"] >= 150]

#the way the dataset's made, there is no column saying whether a given text is AI generated
#or not. Instead for each row, it has both an AI generated response as well as a human
#written response regarding a certain topic. #So to reformat this, we can take the two
#columns that have the AI generated and human written responses

df = df[["wiki_intro", "generated_intro"]]


data = pd.DataFrame()

#and merge them into one column called the intro column
data["intro"] = df["wiki_intro"].tolist() + df["generated_intro"].tolist()

#and then we can just make a new column where the first half of the responses are labelled
#as not AI generated and the second half as AI generated
data["AI Generated"] = ["False" for i in range(len(df))] + ["True" for i in range(len(df))]

#After that we shuffle the dataset and reset the index
data = data.sample(frac=1).reset_index(drop=True)

#now we can start applying text normalization to the responses
lemmatizer = WordNetLemmatizer()

#for each response
for index in data.index:

  text = data.iloc[index]["intro"]

  #use the nltk library to tokenize the text
  tokens = nltk.word_tokenize(text)

  #make everything lowercase
  tokens = [token.lower() for token in tokens]
  #take out all punctuation and stopwords
  tokens = [token for token in tokens if (token not in punctuation) and (token not in stopwords)]

  #and then lemmatize each word
  tokens = [(lemmatizer.lemmatize(word)) for word in tokens]

  #and finally we can join the tokens all back into one string
  text = " ".join(tokens)

  data.at[index, "intro"] = text

#use the Keras library's tokenizer class to encode our tokens into vectors
#this also applies some additional text normalization but 
#a lot of it is redundant

#this will use the 5000 most common words in the dataset and build a frequency 
#vector for each response
tokenizer = Tokenizer(num_words= 5000, lower=True, split=" ")

tokenizer.fit_on_texts(data['intro'].values)

train = tokenizer.texts_to_sequences(data['intro'].values)

train = pad_sequences(train)

model = keras.Sequential()

#after that we set up the architecture of our neural network 
model.add(keras.layers.Embedding(5000, 32, input_length= train.shape[1]))
model.add(keras.layers.LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(keras.layers.LSTM(32, dropout=0.3, recurrent_dropout=0.2))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

labels = pd.get_dummies(data['AI Generated']).values

#split our dataset into a training and testing set
trainx, testx, trainy, testy = train_test_split(train, labels, test_size=0.1, random_state=0)

#and then feed it into our neural network
model.fit(trainx, trainy, epochs = 7, batch_size = 16, verbose=2)

model.save("Classifier_Model")

with open("tokenizer.pickle", "wb") as handle:
  pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
  

#once the model has finished training, we then have it predict whether each response
#in the testing set is AI generated or not
predictions = model.predict(testx)

#and after getting the predictions, we then compute the accuracy of our model
predictions = [0 if pred[0] > pred[1] else 1 for pred in predictions]
ytest = [0 if yval[0] > yval[1] else 1 for yval in testy]
diff = sum([1 if predictions[index] != ytest[index] else 0 for index in range(len(ytest))])

accuracy = 1 - diff/len(predictions)

#Finally to better visualize our results, I also displayed a confusion matrix for the classifier
#and also displayed some relevant statistics like the overall accuracy, the false positive rate,
#and the false negative rate

print("-------------CONFUSION MATRIX FOR CLASSIFIER-------------")
cm = confusion_matrix(ytest, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["AI Generated", "Human Written"])
disp.plot()
plt.show()
print("\n\n")
print("OVERALL ACCURACY: " + str(round(accuracy, 4)*100) + "%")
print("FALSE NEGATIVE RATE: " + str(round(cm[1][0]/(cm[0][0] + cm[1][0]), 4)))
print("FALSE POSITIVE RATE: " + str(round(cm[0][1]/(cm[0][1] + cm[0][0]), 4)))
print("\n\n")


  



