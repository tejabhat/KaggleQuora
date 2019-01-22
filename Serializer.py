# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:21:43 2019

@author: tejbhat

Saves and loads a given object using pickle.
"""

import pickle

class Serializer():    
    def save(self,obj,filename):        
        print("saving to "+ filename)
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self,filename):        
        with open(filename, 'rb') as handle:
            obj = pickle.load(handle)
        return obj
        