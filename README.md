# KaggleQuora
Quora Insincere Questions Classification

This project gives the solution for the Kaggle Competetion - "https://www.kaggle.com/c/quora-insincere-questions-classification".

Brief on Data: 
Train data contains question id, question text and classification-whether the question is insincere or not..
Test Data contains question id, question text.

Total training records - approximately 1306122, 
Insincere questions - approximately 80810.
              

Stages involved in Training:
1. Normalizing the Data:
   This is to help in extracting the important features from the dataset.
   
   a) Questions are POS tagged with nltk libary and person/org names are replaced with a common word.
   This is done as we don't want the model to decide based on a person/org name. ===> This step is not used currently as it takes a long time.
   
   b) If there are negation words, like "wouldn't", transform it as "would not".
   
   c) Remove the stop words as we don't want these to be part of our model. ==> This step is commented out as many of the words like "not" is also a stop word.
   
   d) Remove non-alphabetic characters
   
2) Pre Processing
   a) Convert the word sequence into a number sequnce. 
   I am using Kera library for this, as, in the future we might want to do some customizations like - create the same sequence numbers for train data and test data even if the test run is executed after the application restart.
   
   b) Created the word vectors using gensim library on the corpus containg the words from both train set and test set.
   I didn't use Google or other Embeddings - this may be comprehensive when compared to the available words on the net, but, still misses many of the words from our train/test set. 
   Create an embedding matrix based on both train and test word corpus
   
   
   
3) Training

Uses Neural Network.
