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
   Class 'SentencePreProcessor' does most of the work required for this.
   
   a) Questions are POS tagged with nltk libary and person/org names are replaced with a common word.
   This is done as we don't want the model to decide based on a person/org name.
   
   b) If there are negation words, like "i wouldn't create", transform it as "i not_create".
   
   c) Remove the stop words as we don't want these to be part of our model.
   
2) Pre Processing
   a) Convert the word sequence into a number sequnce. 
   We have created our own class 'WordSequencer' for this, as we would want to create the same sequence numbers for train data and test data even if the 
   test run is executed after the application restart.
   
   b) uses Google Embeddings, and create an embedding matrix based on both train and test word corpus
   
   
   
3) Training

Uses Neural Network.
