# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:54:37 2019

@author: tejbhat
"""

import gensim
import numpy as np

from utils import myprint

class MyWord2Vec():
    
    def load_word2vec(self): 
    
        #ffile="embeddings\\glove.840B.300d.txt"
        ffile="..\\embeddings\\GoogleNews-vectors-negative300.bin"
        #ffile="embeddings\\paragram_300_sl999.txt"
        #ffile="embeddings\\wiki-news-300d-1M.vec"
        
        print("loading embedding word2vec file ",ffile)
        
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(ffile, binary=True)  
        self.EMBEDDING_DIM=300
        

    def build_embed_matrix(self,lword2idx):
        # prepare embedding matrix
        myprint('Filling pre-trained embeddings...')
        
        #index in word2idx  starts from 1.
        embedding_matrix = np.zeros((len(lword2idx)+1, self.EMBEDDING_DIM))
        
        for word, i in lword2idx.items():
            
            try:
                if word.startswith("not_"):
                    word=word.split("_")[1]
                    embedding_vector = self.word2vec[word]
                    embedding_vector= embedding_vector * -1
                else:    
                    embedding_vector = self.word2vec[word]
                
                embedding_matrix[i] = embedding_vector
            except:# Exception as e:
                #print (e)
                #if word is not found in word2vec - 
                #in the zero array, replace few elements with the word index.
                #in that way, we will be creating a unique vector for each word
                #even if the word is not in the embedding dictionary.
                embedding_vector=np.zeros(self.EMBEDDING_DIM)
                for indx in range(0,50):
                    embedding_vector[indx]=i
                #also that partuclar position
                #embedding_vector[i]=i
                embedding_matrix[i]=embedding_vector
                
        return embedding_matrix    