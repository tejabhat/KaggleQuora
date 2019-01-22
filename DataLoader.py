# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:29:01 2019

@author: tejbhat
"""

import pandas as pd
import numpy as np

#imports from this project.
from utils import myprint
from SentencePreProcessor import SentencePreProcessor

class DataLoader():
    def __init__(self):
        self.tokenizer = SentencePreProcessor()
    
    def get_train_test_tokens(self):
        questions_train=self.get_questions("../data/normalized/train_norm.csv")
        myprint("questions train =",len(questions_train))
    
        questions_test=self.get_questions("../data/normalized/test_norm.csv")
        myprint("questions test =",len(questions_test))
        
        questions=np.concatenate((questions_train,questions_test))
        
        
    
                
        myprint("start tokenization")
        tokens_arr=self.tokenizer.tokenize_lines(questions)
        myprint("end tokenization")
        
        return tokens_arr
        
    def get_questions(self,csv):    
        data=pd.read_csv(csv)
        return data["question_text"]   
    
    def get_questions_target(self,csv):    
        data=pd.read_csv(csv)
        #shuffle
        #data = data.sample(frac=1).reset_index(drop=True)
        lst=[]
        for index,row in data.iterrows():
            lst.append( (row["question_text"],int(row["target"])) )
            
        return lst 

    def get_test_data(self):
        testfile=pd.read_csv("..\\data\\normalized\\test_norm.csv")
        testqns=testfile["question_text"]
        testqids=testfile["qid"].values
        return testqns,testqids
        
    def normalize_file(self,filename,normfilename):
        myprint("normalize file ", filename)
        df=pd.read_csv(filename)
        for i,row in df.iterrows():
            v=self.tokenizer.normalize_line(row["question_text"])
            df.set_value(i,"question_text",v)
            if (i%1000==0):
                myprint(i)
        df.to_csv(normfilename)

#normalize_file("data\\test.csv","data\\normalized\\test_norm.csv")
#normalize_file("data\\train.csv","data\\normalized\\train_norm.csv")
        
        
    def prepare_train_data(self,voc):
        myprint("prepare_train_data")
        lst=self.get_questions_target("..\\data\\normalized\\train_norm.csv")
        
        class_0=[(qn,cls) for (qn,cls) in lst if cls==0]
        class_1=[(qn,cls) for (qn,cls) in lst if cls==1]
        
        myprint("cls0=",len(class_0),",cls1=",len(class_1))
        #cls0= 1225312 ,cls1= 80810
        
        #lst=np.concatenate((class_1,class_0[0:len(class_1)]))
        myprint("total=",len(lst))
        
        #lst=np.concatenate((class_1,class_1,class_1,class_1,class_1,class_1,class_0,class_1,class_1,class_1,class_1,class_1,class_1))
        myprint("started shuffling")
        for i in range (0,1000):
            if i%100 == 0:
                myprint(i)
            np.random.shuffle(lst)
            
    
        myprint("writin to file")
        ff=open("..\\data\\shuffle_train.csv","w")
        for (qn,cls) in lst:
            #print(qn)
            try:
                qn=qn.decode('utf8')
            except AttributeError:
                pass            
            try:
                qn=qn.encode('utf-8')
            except:
                pass
            sstr="{},{}".format(qn,cls)                
            ff.write(sstr)

        ff.close()
        myprint("done to file")
            
        #len of class_1 is 50,000, but class_0 is 5,50,000
        #new_lst=[]
        #lth_cls_1=len(class_1)
        #start=0
        #done=False
        #while (done==False):
        #    end=start+lth_cls_1
        #    if ( end >= len(class_0)):
        #        done=True
        #        end=len(class_0)
        #    if len(new_lst)==0:
        #        new_lst=np.concatenate((class_1,class_0[start:end]))
        #    else:
        #        new_lst=np.concatenate((new_lst,class_1,class_0[start:end]))
                
        #    np.random.shuffle(lst)
        #    start=end
        
        #print("new_list=",len(new_lst))
        #lst=new_lst
        
        tokenizer=SentencePreProcessor()
        myprint("tokenize start")
        tokens,targets=tokenizer.tokenize_train_data(lst)    
        #print (tokens)
        #print(targets)
        myprint("tokenize end")
        
        myprint("start seq from tokens")
        seq=voc.sequences_from_tokens(tokens)
        myprint("end seq from tokens")
        
        myprint("len of seq arr=",len(seq))
        seq=voc.pad_sequences(seq)
        myprint("done pad seq")
        
        return seq, targets
            