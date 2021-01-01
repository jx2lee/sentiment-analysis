import tensorflow as tf
import numpy as np
import os
import pickle
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from random import randint

def transY(array):
    result = 1
    if array[0]==1:
        result = 0
    
    return result

def LoadData(np_path, path_1, path_2):
    
    A = np.load(np_path+path_1)['arr_0']
    b = np.load(np_path+path_2)['arr_0']

    X_train, X_test, y_train, y_test = train_test_split(A, b, test_size=0.2)
    
    return X_train, X_test, y_train, y_test

def arraytoDf(array1, array2):

    seq = []
    seqLen = []
    sentiment = []

    for i in range(len(array1)):
        seq.append(array1[i])
        seqLen.append(len(array1[i]))
        sentiment.append(transY(array2[i]))

    dic = {'seq' : seq, 'seqLength' : seqLen, 'sentiment' : sentiment}
    df = pd.DataFrame(dic, columns=['seq', 'seqLength', 'sentiment'])
    return df

class Data():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n
        return res['seq'], res['sentiment'], res['seqLength']
    
#padding
class PaddedData(Data):
    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['seqLength'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['seqLength'].values[i]] = res['seq'].values[i]

        return x, res['sentiment'], res['seqLength']
    


