# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:29:30 2018

@author: sande
"""
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from Data.datareader import *


#for fivegrams, iterate over all sensors
#F_PU3, F_PU5, F_PU9 have been ignored because they contain only zeros in the dataset
for fields in ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5','L_T6', 'L_T7', 'F_PU1','F_PU2', 'F_PU4', 'F_PU6', 'F_PU7', 'F_PU8', 'F_PU10','F_PU11', 'F_V2', 'S_V2', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415','P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']:
        print ("for sensor", fields)
        series = read_3_series(fields)
        #and for each sensor, discretize the data using different number of quantiles
        for no_quantiles in [5]:
            _, bins = pd.qcut(series, no_quantiles, retbins=True, duplicates = 'drop')
            cut_data, bins = pd.qcut(series, no_quantiles, retbins=True, labels=range(len(bins)-1),duplicates = 'drop')
            
            #create a matrix of dimensions of no_quantiles
            occurrences_train = np.ones((no_quantiles,no_quantiles,no_quantiles,no_quantiles,no_quantiles))
            sum = 0
            #try for five-grams
            for i in range(0,len(cut_data)-5): 
                a = cut_data[i]
                b = cut_data[i+1]
                c = cut_data[i+2]
                d = cut_data[i+3]
                e = cut_data[i+4]
                occurrences_train[a][b][c][d][e] = occurrences_train[a][b][c][d][e] + 1
                
            #convert to probabilities
            occurrences_train_prob = np.true_divide(occurrences_train, len(cut_data))
            #For test data, evaluate the performance
            series = read_4_series(fields)
            #print (occurrences_train_prob)
            _, bins = pd.qcut(series, no_quantiles, retbins=True, duplicates = 'drop')
            #no_quantiles = len(bins)
            cut_data, bins = pd.qcut(series, no_quantiles, retbins=True, labels=range(len(bins)-1), duplicates = 'drop')
    

            
            for threshold in [0.005] :
                count = 0
                true_positive_count = 0
                for i in range(0,len(cut_data)-5): 
                    a = cut_data[i]
                    b = cut_data[i+1]
                    c = cut_data[i+2]
                    d = cut_data[i+3]
                    e = cut_data[i+4]
                    if (occurrences_train_prob[a][b][c][d][e] < threshold):
                        #Count the number of five grams which are under threshold
                        count = count + 1
                        if attack_at_time_04(cut_data.index[i+4]) :
                            #If there is an attack at the time of the fifth entry of five grams, then it is true positive 
                            true_positive_count = true_positive_count + 1
                
                if ((count > 0) and (true_positive_count > 0)) :
                    print ("Precision: ",true_positive_count/count," for threshold: ", threshold," and number of percentiles: ",no_quantiles)
                    
