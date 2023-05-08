# AI-Text-Detection
This is my final project for ECE-467: Natural Language Processing. I built a binary classification model that predicts whether 
or not a given text was written by openAI. I have also included an interface that allows you to enter in a text and get the 
probability that the text was written by openAI. 



Instructions on Running the Program
I developed my system in version 12.3.1 of the macOS Monterey operating system. To run the program you need python 
3.9.12 or a later version installed. You will also need to install the following libraries in order to run the program: pandas, 
numpy, sklearn, nltk, keras, and matplotlib. You can install these libraries by entering the following command into your 
terminal: pip3 install <package name>. 
	There are two possible programs that can be run. The first one is called Khan_F_ECE_467_Final_Project.py. To run this 
program, you need to download the GPT-wiki-intro dataset which can be found at the following link: 
https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro. Place the python file and this dataset in the same directory and 
the model will begin to train on the dataset. This program will take roughly 7 hours to complete. Once it’s complete you will be 
able to see how the model performed on the training and testing set as well as a confusion matrix that gives more detail on how 
the model performed on the testing detail.
	The other program that can be run is called AIDetection.py. To run this program, you need to have the Classifier_Model 
folder and this script in the same directory. This folder contains the classifier that I had trained for this project and the 
program provides an interface through which you can enter essays that you believe are written by an AI and the model will 
determine the likelihood that this was the case. When you wish to exit the program, you can just type the word exit and the 
program will terminate.

Implementation Details and Performance Evaluation

	My first obstacle in creating my classifier was finding a good dataset to work off from. Unfortunately while there are 
many classifiers like the one I was looking for, the datasets that they use aren’t public so this ended up being challenging. 
Luckily however, I found an incredibly good dataset called the GPT-wiki-intro dataset that does exactly this. Each entry in this 
dataset was created by prompting openAI for a “200 word wikipedia style introduction” on a certain topic and then also 
extracting the corresponding introduction paragraph from wikipedia which we can assume to be human generated. Altogether this 
dataset contains over 150,000 introductions taken from both openAI and Wikipedia so 300,000 samples of data which is incredibly 
large.
	Training on the entire dataset takes around 7 hours so experimenting with different parameter values using the entire 
dataset would have taken a very long time. Instead, I decided to use around 1/50th of the dataset to try out different 
parameters. This reduced the size of my dataset to around 6000 samples. One concern might be that making the dataset this small 
would make the variance of my accuracy on the testing set greater so that one run of the script wouldn’t tell me anything 
definitive about the advantages of using one set of parameters over another. However, 6000 samples is still fairly large so this 
shouldn’t be an issue. And since 


Doing this reduced my training time to roughly 25 minutes which is still fairly long but manageable.
One thing I found was that the model typically has trouble classifying short texts that are less than 150 words. When responses 
weren’t filtered out based on their length, the model achieved an accuracy of 82.43 %. In my opinion, this isn’t an accurate 
depiction of the model’s accuracy because ideally it will be used to classify documents like essays and emails which would 
typically have more than 150 words. In this dataset, such responses make up roughly ⅔ of the AI generated responses and none of 
the responses taken from Wikipedia. I didn’t want to introduce such a large skew in the dataset so I decided to take out the 
Wikipedia response as well if the corresponding AI generated response wasn’t long enough. This left me with around 100,000 
responses which is still fairly large. To make the comparison to the model trained on the unfiltered model more fair, I used 
6000 samples that all had responses greater than 150 words so that the two models would have had the same amount of data to 
train on. Once I filtered out short responses, the classifier’s accuracy jumped to 87.21%.
After that, I applied some traditional text processing techniques like making all words lowercase, performing lemmatization, and 
removing punctuation and stop words. I was concerned that removing stop words would degrade the performance of the model since 
it’s possible that frequency of the usage of certain stopwords might tell us something about openAI’s style of writing. However, 
removing stopwords didn’t really affect the performance of the classifier so I decided to remove them.
For implementing the neural network, I first used the Tokenizer function in the keras library to encode my response using one 
hot encoding. Since the dataset is so large, I decided to use the 5000 most common words in the training set for the encoding 
and all words outside of the 5000 most common were lumped into the same category. I then used an embedding layer with 32 nodes 
which would significantly reduce the dimension of the input to 32. This is essential because given the number and length of 
these responses, the one hot encoded vector representations of the responses are going to be very sparse which is incredibly 
inefficient since that would mean that the first hidden layer of our neural network would have to be at least 150 nodes long and 
this would likely lead to overfitting as well.

After the embedding layer, I decided to add in two LSTM layers. I wanted to use an RNN for my classifier because of its ability 
to remember long term dependencies which would make it easier for the network to draw on connections found in previous training 
samples. LSTM networks seemed like the best RNN that does this and the only drawback seemed to be that because it's much more 
complex than architectures that use GRU or regular RNNs, its training time is significantly longer. 
With all of these changes, I didn’t really expect the model to do that well however. It didn’t seem right that just knowing the 
order of words in a response would be enough for it to correctly classify responses across literally hundreds of thousands of 
distinct topics. The fundamental difference between AI generated responses and responses made by humans has to be structural. I 
thought that I would have to use something like an ngram representation of the responses using only POS tags. But when I ran my 
model using the former approach, I achieved an accuracy of 97.2% on the training set and 93.8% on the testing set which is 
incredible.

Additionally, from the confusion matrix you can see that the number of false positives and false negatives are very close to one 
another which speaks to how good the dataset that the model trained on was. Not only were there an equal number of AI generated 
texts as there were human written, each topic written by ChatGPT had a corresponding text that was pulled from Wikipedia. This I 
think was critical to the success of the classifier because by having texts of the same topic written by both a human and an AI, 
it can more easily compare and see the differences between the two.

<img width="556" alt="Screen Shot 2023-05-07 at 10 05 19 PM" src="https://user-images.githubusercontent.com/78983433/236717951-d0326505-8f2f-4ac2-9128-314533fc48eb.png">



To test out my model outside of the dataset, I created a separate python file and loaded in the model that I had trained there. 
Then I would prompt the user for a text and I would run it through the model to determine how likely the model thinks it was 
written by openAI. I didn’t do any extensive testing but of the 6 essays that were generated by openAI that I gave to the model, 
it was able to correctly predict AI generated text in 4 of them. The other thing that’s very interesting is that the classifier 
would generally only give a very high probability of a text being AI generated or a very low one, nothing in between.




<img width="696" alt="Screen Shot 2023-05-07 at 10 10 00 PM" src="https://user-images.githubusercontent.com/78983433/236718441-016532f0-257f-4c81-8a84-b14ce000fdb4.png">


