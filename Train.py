# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:43:51 2019

@author: tejbhat
"""

import numpy as np
import torch
import torch.nn as nn


from MyModel import MyModel
from DataLoader import DataLoader

from utils import myprint

class Train:
    
    def __init__(self,EMBEDDING_DIM,embedding_matrix,wordsequencer):
        self.EMBEDDING_DIM=EMBEDDING_DIM
        self.embedding_matrix=embedding_matrix
        self.class_names=[0,1]    
        self.BATCH_SIZE=2000
        myprint('batch size=',self.BATCH_SIZE)
        #loss_fn=torch.nn.MSELoss(reduction="sum")
        self.loss_fn=nn.BCEWithLogitsLoss()
        self.wordsequencer=wordsequencer
        #optim=torch.optim.SGD(mynet.parameters(),lr=1e-4,momentum=0.9)
        
        self.loader=DataLoader()

    def run(self,start_epoch,end_epoch):    
        
        model=MyModel(self.wordsequencer.num_words,self.EMBEDDING_DIM,self.embedding_matrix)

        optim=torch.optim.Adam(model.parameters())
    
        model.train()
        gru_h=None

        for epoch in range(start_epoch,end_epoch):

            sequences_arr,targets=self.loader.prepare_train_data(self.wordsequencer)
        
            data_len=len(sequences_arr)
            print("datalen=",data_len)
        
            epoch_loss=0
            epoch_accuracy=0

            cnt=0
            total=0
            correct=0

            for i in range (0,data_len,self.BATCH_SIZE):
            
                myprint("epoch-",epoch, " batch-",cnt)
        
                end=i+self.BATCH_SIZE
                lth=len(sequences_arr)
                if end > lth:
                    #this last batch is lesser than the standard batch size.
                    #we add the initial question set to make this batch of standard size.
                    batch_input=sequences_arr[i:lth]
                    batch_target=targets[i:lth]

                    #fill the remaining with the questions from first.
                    remain_input=sequences_arr[0:end-lth]
                    remain_target=targets[0:end-lth]

                    batch_input=np.concatenate((batch_input,remain_input))
                    batch_target=np.concatenate((batch_target,remain_target))

                else:            
                    batch_input=sequences_arr[i:end]
                    try:
                        batch_target=targets[i:end]
                    except Exception as e:
                        print(str(e))
                        print(i,end)
                        print(targets[i:end])
                
            
                class_0=[cls for cls in batch_target if cls==0]
                class_1=[cls for cls in batch_target if cls==1]
                print("len cls0=",len(class_0)," cls1=",len(class_1))
            
                batch_target=torch.Tensor(batch_target)
                #for rnn - we want the complete sentence to be covered by one column.
                #this enables catching the sequencing of words.
                #so, transpose the horizontal sentence(words) arr into a column array.
    
    
                #rnn_batch_input=torch.LongTensor(np.transpose(cnn_batch_input))
                batch_input=torch.LongTensor(batch_input)

            
                optim.zero_grad()
                myprint( "rnn start")
                y_pred,gru_h=model(batch_input,gru_h)
                myprint("rnn loss")
                loss=self.loss_fn(y_pred,batch_target)
                
                myprint("rnn backward")
                loss.backward()
                optim.step()

                #print("gru_h before=",gru_h)
                myprint("detach gru_h")
                gru_h.detach_()
                #print("gru_h after=",gru_h)
                myprint("rnn optim step")            
            
                epoch_loss+=loss.item()
            
                y_pred_rounded=model.round_prediction(y_pred)
            
                class_0=[cls for cls in y_pred_rounded if cls==0]
                class_1=[cls for cls in y_pred_rounded if cls==1]
                print("predicted len cls0=",len(class_0)," cls1=",len(class_1))
            
                y_pred_sigmoid= torch.sigmoid(y_pred)
                acc,num_correct = model.binary_accuracy(y_pred_rounded, batch_target)
                epoch_accuracy += acc.item()
                correct+=num_correct
                        
                myprint( "epoch-",epoch," batch=", cnt,
                  " loss=",loss.item()," accuracy=", acc,
                  ", correctly predicted =",num_correct)
            
                model.plot_prediction( y_pred,y_pred_rounded)
                model.plot_metrics(y_pred_rounded, y_pred_sigmoid.detach().numpy(),batch_target)


            
                cnt=cnt+1        
                total+=len(batch_target)            

            epoch_loss=epoch_loss / cnt
            epoch_accuracy= epoch_accuracy / cnt

            myprint( "epoch-",epoch," num_batches=", cnt,
              " loss=",epoch_loss," accuracy=", epoch_accuracy, 
              " total=",total,
              ", correctly predicted=",correct)
    
        torch.save(model.state_dict(), "model.pth")
        return model, optim, self.loss_fn,gru_h
    
