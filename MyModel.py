# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:13:44 2019

@author: tejbhat
"""
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from Confusion import Confusion
from utils import myprint

class MyModel(nn.Module):    

    def create_embed_layer(self,numwords, EMBEDDING_DIM, embedding_matrix):    
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
    
        #torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, 
        #sparse=False, _weight=None)
        embedding_layer = nn.Embedding(
          numwords+1,
          EMBEDDING_DIM)
    
        embedding_layer.load_state_dict({'weight': torch.FloatTensor(embedding_matrix)})
        ## or do this - embedding_layer.weight.data.copy_(pretrained_embeddings)
    
        #we do not want to update the pretrained embeddings. so set requires_grad to False.
        embedding_layer.weight.requires_grad = False
        return embedding_layer


    
    def __init__(self,numwords,EMBEDDING_DIM,embedding_matrix):
        super(MyModel,self).__init__()
        
        op_classes=1
        
        self.EMBEDDING_DIM=EMBEDDING_DIM
        self.embedding=self.create_embed_layer(numwords,EMBEDDING_DIM,embedding_matrix)
        
        #considers bigrams
        self.conv1 = nn.Conv2d(in_channels=1, 
                                out_channels=EMBEDDING_DIM, 
                                kernel_size=(2,EMBEDDING_DIM))

        self.gru_hidden_size=30        
        self.gru_num_layers=3
        
        ### NOW CREATE A GRU BI DIRECTIONAL NETWORK, with a small dropout.
        self.gru = nn.GRU(EMBEDDING_DIM, self.gru_hidden_size, num_layers=self.gru_num_layers, bidirectional=True, 
                           dropout=0.1)
        
        #self.rnn=nn.RNN(EMBEDDING_DIM,self.hidden_nodes)
        #self.rnn=nn.LSTM(EMBEDDING_DIM,self.hidden_nodes,num_layers=n_layers,bidirectional=self.bidirectional,dropout=dropout)
        
        
        #from  bidirectional LSTM - we will consider the last two layers - ie., last forward layer and last backwar layer output.
        #that's why the input to next layer will become hidden_nodes*2
        self.fc=nn.Linear(self.gru_hidden_size*2,op_classes)
        
        self.figcnt=0
        self.class_names=[0,1]
        self.confusion=Confusion()
        
        
    def forward(self, input,gru_h):        
        #[batch_size x WORD_SEQ_IN_SENTENCE]
        embed=self.embedding(input)        
        #[batch size, sentence length, emb dim]
        #packed=torch.nn.utils.rnn.pack_padded_sequence(embed2,input_lengths)
        
        ############# CONVOLUTIONAL LAYER ##################
        #x = embed.unsqueeze(1)
        #[batch size, 1, sentence length, emb dim]
        
        #conv1=self.conv1(x)
        
        #conv1=conv1.squeeze(3)
        #conv1 = F.relu(conv1)
        #conv1 = F.max_pool1d(conv1, 2)
        #conv1=conv1.view(sh[2],sh[0],sh[1])
        ######## RNN #######################
        conv1=embed
        sh=conv1.shape
        conv1=conv1.view(sh[1],sh[0],sh[2])
        #sent_lth_after_max_pool x batchsize x hidden_size
        print("conv1=",sh)
        print("input=",conv1.shape)
        #num_layers * num_directions, batch, hidden_size
        #set the hidden state for the first time.
        #if gru_h is None:
        #    gru_h=torch.zeros((gru_num_layers*2,conv1.shape[1],gru_hidden_size))
            
        output,gru_h=self.gru(conv1,gru_h)        
        #output = [sent len, batch/sample size, hid dim]
        #hidden = [1, batch size, hid dim]
        print("hidden -2 ", gru_h[-2,:,:].size())
        print("hidden -1 ", gru_h[-1,:,:].size())
        
        #gru_hidden_state is of shape (num_layers * num_directions, batch, hidden_size): 
        #if we have 2 layers, (-1,..) gives the output from the previous run
        # and  (-2,..) gives the output from the previous-to-previous run.
        
        ##concat the data from the last forward layer and the last backward layer.
        cat=torch.cat((gru_h[-2,:,:],gru_h[-1,:,:]),dim=1)
        
        print("gru_h concatenated - shape=",cat.shape)
        #following is necessary for training the last batch which may be lesser than the standard batch_size chosen
        #to avoid "RuntimeError: Expected hidden size (4, 1122, 256), got (4, 5000, 256)".
        #this_batch_size=input.shape[1]
        #cat=cat[:,-this_batch_size:,:]
        
        out = self.fc(cat)
        print("out=",out.shape)
        out=out.squeeze(1)    
        print("out afte squeeze=",out.shape)
        y_pred=out
        return y_pred,gru_h

    def round_prediction(self,preds):    
        #round predictions to the closest integer
        preds = torch.sigmoid(preds)
        preds=np.array([0 if i<0.7 else 1 for i in preds])
        return preds
        
    def binary_accuracy(self,rounded_preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        #round predictions to the closest integer
        
        correct = (torch.Tensor(rounded_preds) == y).float() #convert into float for division     
        acc = correct.sum()/len(correct)
        return acc, correct.sum().item()
    

    def plot_metrics(self,y_pred_rounded,y_pred,batch_target):                        

        # Plot non-normalized confusion matrix
        plt.figure(self.figcnt)
        self.figcnt+=1
        cm=self.confusion.calc(batch_target,y_pred_rounded) 
        self.confusion.plot(cm, classes=self.class_names,
              title='Confusion matrix, without normalization')

        fpr, tpr, thresholds = roc_curve(batch_target, y_pred_rounded, pos_label=2)
        myprint("fpr=",fpr,", tpr=",tpr,",thresholds=",thresholds)
    

    def plot_prediction(self,y_pred,y_pred_rounded):
        y=[]
        cnt=1
        for i in y_pred:
            y.append(cnt)
            cnt=cnt+1 
    
        plt.figure(self.figcnt)
        plt.title("sigmoid_{}".format(self.figcnt))
        plt.scatter(y,y_pred,s=5)
        plt.show()
        self.figcnt+=1
                        
        plt.figure(self.figcnt)
        plt.title("rounded_{}".format(self.figcnt))
        plt.scatter(y,y_pred_rounded,s=5)
        plt.show()
        self.figcnt+=1
           
    
