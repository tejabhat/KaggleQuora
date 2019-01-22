# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 08:50:04 2019

@author: tejbhat

during trianing: Writes the "config.ini" file that contains the MAX_SENTENCE_LENGTH .
during test run of the model: reads the config.ini file. 
"""
import configparser
class Config():
    def write(self,ddict):
        config=configparser.ConfigParser()
        try:
            config.read("config.ini")
        except:            
            a=1#ignore
            
        if 'default' not in config:
            config['default']={}

        for k,v in ddict.items():
            config['default'][k]=v
            
        ff=open("config.ini","w")
        config.write(ff)
        ff.close()


    def read(self,arr):
        ret={}
        config=configparser.ConfigParser()
        try:
            config.read("config.ini")
            if "default" in config:
                for k in arr:
                    ret[k]=config['default'][k]
        except:
            a=1#ignore
        return ret