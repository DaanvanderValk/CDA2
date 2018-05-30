# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:13:58 2018

@author: sande
"""

from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from Data.datareader import *

fields = ['DATETIME','L_T1']

# Read a series
#series = read_csv('../Data/BATADAL_dataset03.csv', parse_dates=[0], index_col=0, squeeze=True,usecols=fields)
series = read_3_series('L_T1')

no_quantiles = 5

cut_data, bins = pd.qcut(series, no_quantiles, retbins=True, labels=range(no_quantiles))

#create a matrix of dimensions of no_quantiles
#occurrences_train = np.ones((no_quantiles,no_quantiles,no_quantiles,no_quantiles,no_quantiles))
occurrences_train = np.ones((no_quantiles,no_quantiles,no_quantiles))
sum = 0
#try for tri-grams
for i in range(0,len(cut_data)-5): 
    a = cut_data[i]
    b = cut_data[i+1]
    c = cut_data[i+2]
#    d = cut_data[i+3]
#    e = cut_data[i+4]
    #occurrences_train[a][b][c][d][e] = occurrences_train[a][b][c][d][e]+1
    occurrences_train[a][b][c] = occurrences_train[a][b][c]+1

#print (occurrences_train)

#convert to probabilities
occurrences_train_prob = np.true_divide(occurrences_train, len(cut_data))
#print (occurrences_train_prob)    

#fields = ['DATETIME','L_T1']
# Read a series
#series = read_csv('../Data/BATADAL_dataset04.csv', parse_dates=[0], index_col=0, squeeze=True,usecols=fields)
series = read_4_series('L_T1')

cut_data, bins = pd.qcut(series, no_quantiles, retbins=True, labels=range(no_quantiles))
print (cut_data.head())
print (bins)
#occurrences_test = np.ones((no_quantiles,no_quantiles,no_quantiles,no_quantiles,no_quantiles))
occurrences_test = np.ones((no_quantiles,no_quantiles,no_quantiles))
sum = 0
for i in range(0,len(cut_data)-5): 
    a = cut_data[i]
    b = cut_data[i+1]
    c = cut_data[i+2]
#    d = cut_data[i+3]
#    e = cut_data[i+4]
    #if (occurrences_train_prob[a][b][c][d][e] < 0.0002):
    if (occurrences_train_prob[a][b][c] < 0.0005):
        #print (pd.value_counts(cut_data))
        #print (occurrences_train_prob[a][b][c][d][e])
        print (occurrences_train_prob[a][b][c])
        print (cut_data.index[i])

#occurrences_test_prob = np.true_divide(occurrences_test, len(cut_data))
#print (occurrences_test_prob)
#
#print (occurrences_test_prob.shape)
#print (occurrences_train_prob.shape)
#for i in range(0,4):
#    for j in range(0,4):
#        for k in range(0,4):
#            #if ((occurrences_test_prob[i][j][k] - occurrences_train_prob[i][j][k])>0.001):
#            if ((abs(occurrences_test_prob[i][j][k] - occurrences_train_prob[i][j][k])) > 0.001):
#                print ("Diff found at index ",i,j,k)
#                print (occurrences_test_prob[i][j][k])
#                print (occurrences_train_prob[i][j][k])
#
