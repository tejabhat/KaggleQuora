# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:00:03 2019

@author: tejbhat
"""

import pandas as pd
import numpy as np
import torch

from SentencePreProcessor import SentencePreProcessor
from utils import myprint

from DataLoader import DataLoader

class Test():
    
    def prepare_data(self,voc):
        loader=DataLoader()
        testqns,testqids=loader.get_test_data()
        
    
        #THIS IS REQUIRED IN THE KERNEL, but ocmmenting here as we
        #have already normalized the test data
        #myprint("start norm")
        #normqns=tokenizer.normalize_lines(testqns)
        #myprint("end norm")
            
        #myprint("start tokenization")
        #testtokens=tokenizer.tokenize_n_gram(normqns,3)
        #myprint("end tokenization")
        
        #testqns=[testqns[len(testqns)-1]]
        #testqids=[testqids[len(testqids)-1]]
        #print(testqns, testqids)
        tokenizer=SentencePreProcessor()
        myprint("start tokenization")
        testtokens=tokenizer.tokenize_lines(testqns)
        myprint("end tokenization")
        
        myprint("start seq from tokens")
        testseq=voc.sequences_from_tokens(testtokens)
        myprint("end seq from tokens")
    
        myprint("len of seq arr=",len(testseq))
        testseq=voc.pad_test_sequences(testseq)
        myprint("done pad seq")
        
        return testseq, testqids


    def run(self,testseq,testqids,model,optim,gru_h):
        
        model.eval()
    
        ffile=open("test_result.csv","w")
        ffile_raw=open("test_raw_result.csv","w")
        
        testseq_len=len(testseq)
        print(testseq_len)
        BATCH_SIZE=2000
        for j in range (0,testseq_len,BATCH_SIZE):
            end=j+BATCH_SIZE
            if end < testseq_len:            
                batch_input=testseq[j:end]
                batch_qid=testqids[j:end]
            else:
                batch_input=testseq[j:testseq_len]
                batch_qid=testqids[j:testseq_len]
                
                ## make up the batchsize by repeating the last question.
                last_input=testseq[testseq_len-1]
                last_qid=testqids[testseq_len-1]
                for i in range(testseq_len,end):
                    batch_input.append(last_input)
                    batch_qid=np.append(batch_qid,last_qid)
                    
            

        myprint("batchinput=",len(batch_input),"batch_qd=",len(batch_qid))
        #for rnn - we want the complete sentence to be covered by one column.
        #this enables catching the sequencing of words.
        #so, transpose the horizontal sentence(words) arr into a column array.
        #rnn_batch_input=torch.LongTensor(np.transpose(batch_input))
        rnn_batch_input=torch.LongTensor(batch_input)

        optim.zero_grad()

        with torch.set_grad_enabled(False):
            #myprint("rnn start")
            preds,_=model(rnn_batch_input,gru_h)
            #print(preds)
            #myprint("rnn end")
            myprint("rnn batch input=",rnn_batch_input.shape, "preds=",len(preds))
            sigmoid_preds=torch.sigmoid(preds)
            rounded_preds=model.round_prediction(preds)
            model.plot_predictoin(sigmoid_preds,rounded_preds)            
            print(len(batch_qid), len(rounded_preds))

            cnt=0
            for qid in batch_qid:
                lline="{},{}\n".format(qid,rounded_preds[cnt])            
                ffile.write(lline)
                
                lline="{},{}\n".format(qid,sigmoid_preds[cnt])
                ffile_raw.write(lline)
                cnt+=1

        ffile.close()
        ffile_raw.close()
    
    