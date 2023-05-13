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


model = keras.models.load_model("Classifier_Model")


with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

def processText(essay):
    lemmatizer = WordNetLemmatizer()
    #use the nltk library to tokenize the text
    tokens = nltk.word_tokenize(essay)
    #make everything lowercase
    tokens = [token.lower() for token in tokens]
    #take out all punctuation and stopwords
    tokens = [token for token in tokens if (token not in punctuation) and (token not in stopwords)]

    #and then lemmatize each word
    tokens = [(lemmatizer.lemmatize(word)) for word in tokens]

    #and finally we can join the tokens all back into one string
    text = " ".join(tokens)

    df = pd.DataFrame()
    df["intro"] = [text]

    text = tokenizer.texts_to_sequences(df["intro"].values)

    
    text = pad_sequences(text, maxlen = model.layers[0].input_shape[1])

    
    return text
      
    


while(True):
    essay = input("Enter a text that you suspect might be AI Generated\n\n")

    if essay == "exit":
        print("Goodbye!")
        break
    
    essay = processText(essay)
    prediction = model.predict(essay)
    print("Our classifier predicted that there's a " + str(round(prediction[0][1], 3)*100) + "%"  + " chance that this text was AI Generated")

    print("\n\n")   
