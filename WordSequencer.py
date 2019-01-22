# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:24:27 2019

@author: tejbhat

We have our own sequencer to have more control and we can train on both train and test words
and save using pickle and reuse the same sequences any time during test sequences.
"""

import torch
import torch.nn.functional as F

import numpy as np

#Classes from this project
from Config import Config

class WordSequencer():
    def __init__(self):
        self.word2index={} #SOS_token:0, EOS_token:1}
        self.word2count={}
        self.index2word={}#0:SOS_token,1:EOS_token}
        self.num_words=0#2#SOS,EOS,PAD
        
    def addWord(self,word):
        #if word in (SOS_token,EOS_token):
        #    return
        
        if word not in self.word2index:
            self.word2index[word]=self.num_words            
            self.word2count[word] = 1
            self.index2word[self.num_words]=word
            self.num_words+= 1
        else:
            self.word2count[word] += 1
    
    def fit_to_tokens(self,arr_tokens):
        for arr in arr_tokens:
            for word in arr:
                self.addWord(word)                    
    
    def sequences_from_tokens(self,tokens_arr):
        ret=[]
        for arr in tokens_arr:
            ind_arr=[]
            for word in arr:
                ind_arr.append(self.word2index[word])
            ret.append(ind_arr)
        return ret
        
    def pad_sequences(self,my_seq_arr):
        #we want to take the max_sent_lth as the length of the 95% data.
        lths=[len(seq) for seq in my_seq_arr]
        lths=np.sort(lths)    
        mmax=lths[-1]         
        mmax_95_perc=lths[int(len(lths)*0.95)]
        diff=mmax-mmax_95_perc
        mmax_5_perc=lths[int(len(lths)*0.05)]
        print(mmax,mmax_95_perc,mmax_5_perc," diff=",diff)
        if diff > mmax_5_perc:    
            MAX_SENTENCE_LENGTH=mmax_95_perc
        else:
            MAX_SENTENCE_LENGTH=mmax

        print("MAX_SENTENCE_LENGTH=",MAX_SENTENCE_LENGTH)
        my_seq_arr=[F.pad(torch.FloatTensor(s),(0,MAX_SENTENCE_LENGTH-len(s)),"constant",0).numpy()[:] for s in my_seq_arr]
        
        #save this, to reuse during test cycle.
        config=Config()
        config.write({"max_sentence_length":"{}".format(MAX_SENTENCE_LENGTH)})

        return my_seq_arr

    def pad_test_sequences(self,my_seq_arr):
        config=Config()
        ddict=config.read(["max_sentence_length"])
        max_lth=int(ddict["max_sentence_length"])

        my_seq_arr=[F.pad(torch.FloatTensor(s),(0,max_lth-len(s)),"constant",0).numpy()[:] for s in my_seq_arr]
        return my_seq_arr    