# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:13:05 2019

@author: tejbhat
"""
from datetime import datetime


def myprint(*argv):
    sstr=datetime.now()
    for arg in argv:
        sstr="{} {}".format(sstr,arg)
    print(sstr)
    
    