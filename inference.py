# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:54:59 2019

@author: dengxy
"""
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
import random
#from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle   

#convert one-hot encoding to the label
def one_hot_to_label(label):
    y = []
    for data in label:
        if data[0] == 0:
            l = 1
        else:
            l = 0
        y.append(l)
    return y

#convert softmax out to the lable
def props_to_onehot(softmax_out, therod): 

    for data in softmax_out:
        if data >= therod:
            data = 1
        else:
            data = 0
    
    return softmax_out
    

#random batch
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size) 
    X_batch = X_train[rnd_indices] 
    y_batch = y_train[rnd_indices] 
    return X_batch, y_batch

#label to one-hot encoding
def label_change(befor_label):
    ohe = OneHotEncoder(categories='auto')
    ohe.fit([[0],[1]])
    label = ohe.transform(befor_label).toarray()
    
    return label


def readDataSet(dataSet):
    #data = []
    #print("Loading dataSet......")
    data = np.loadtxt(dataSet, dtype = 'float32', delimiter = ' ')
#    data = shuffle(data)
    return data

def train_data(data):

    receptor = data[:, :n_input]
    ligand = data[:, n_input + 1 : -1]
    label_tmp = data[:, -1].reshape(-1, 1)
    label = label_change(label_tmp)
    
    train_x = np.hstack([receptor, ligand])
    
    return train_x, label


n_input = 217

