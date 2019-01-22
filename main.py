# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:36:10 2019

@author: tejbhat
"""

#Sree Ganeshaaya Namaha
from MyWord2Vec import MyWord2Vec
from DataLoader import DataLoader
from WordSequencer import WordSequencer
from Serializer import Serializer
from Train import Train

from utils import myprint

serializer=Serializer()

voc=WordSequencer()
loader=DataLoader()

arr=loader.get_train_test_tokens()
voc.fit_to_tokens(arr)

myprint("numwords=",voc.num_words) 


#serializer.save(voc,"voc")




wvec=MyWord2Vec()

myprint("load word2vec started")
wvec.load_word2vec()
myprint("word2vec done")



#voc=serializer.load("voc")
myprint("embed matrix start")
embedding_matrix=wvec.build_embed_matrix(voc.word2index)
myprint("embed matrix done =",len(embedding_matrix))

train=Train(wvec.EMBEDDING_DIM,embedding_matrix,voc)
train.run(0,1)

