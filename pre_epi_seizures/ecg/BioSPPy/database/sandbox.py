'''
Created on 4 de Dez de 2012

@author: Carlos
'''

# imports
import pymongo
from pymongo import MongoClient
import gridfs
import numpy as np


if __name__ == '__main__':
    # connect to mongo
    connection = MongoClient('193.136.222.234', 27017)
    db = connection['syncTest']
    
    # gridfs instance
    fs = gridfs.GridFS(db)
    
    # put
    a = fs.put("hello world")
    
    # get
    fid = fs.get(a)
    print fid.read()
    