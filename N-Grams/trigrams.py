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


#for trigrams, iterate over all sensors
#F_PU3, F_PU5, F_PU9 have been ignored because they contain only zeros in the dataset
for fields in ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5','L_T6', 'L_T7', 'F_PU1','F_PU2', 'F_PU4', 'F_PU6', 'F_PU7', 'F_PU8', 'F_PU10','F_PU11', 'F_V2', 'S_V2', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415','P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']:
        print ("for sensor", fields)
        series = read_3_series(fields)
        #and for each sensor, discretize the data using different number of quantiles
        for no_quantiles in [5,10,15,20]:
            _, bins = pd.qcut(series, no_quantiles, retbins=True, duplicates = 'drop')
            cut_data, bins = pd.qcut(series, no_quantiles, retbins=True, labels=range(len(bins)-1),duplicates = 'drop')
            
            #create a matrix of dimensions of no_quantiles
            occurrences_train = np.ones((no_quantiles,no_quantiles,no_quantiles))
            sum = 0
            #try for tri-grams
            for i in range(0,len(cut_data)-5): 
                a = cut_data[i]
                b = cut_data[i+1]
                c = cut_data[i+2]
                occurrences_train[a][b][c] = occurrences_train[a][b][c] + 1
                
            #convert to probabilities
            occurrences_train_prob = np.true_divide(occurrences_train, len(cut_data))
            #For test data, evaluate the performance
            series = read_4_series(fields)
            #print (occurrences_train_prob)
            _, bins = pd.qcut(series, no_quantiles, retbins=True, duplicates = 'drop')
            #no_quantiles = len(bins)
            cut_data, bins = pd.qcut(series, no_quantiles, retbins=True, labels=range(len(bins)-1), duplicates = 'drop')
    

            
            for threshold in [0.0001, 0.0002, 0.0005, 0.001, 0.005]:
                count = 0
                true_positive_count = 0
                for i in range(0,len(cut_data)-5): 
                    a = cut_data[i]
                    b = cut_data[i+1]
                    c = cut_data[i+2]
                    if (occurrences_train_prob[a][b][c] < threshold):
                        #Count the number of tri grams which are under threshold
                        count = count + 1
                        if attack_at_time_04(cut_data.index[i+2]) :
                            #If there is an attack at the time of the third entry of tri grams, then it is true positive 
                            true_positive_count = true_positive_count + 1
                
                if ((count > 0) and (true_positive_count > 0)) :
                    print ("Precision: ",true_positive_count/count," for threshold: ", threshold," and number of percentiles: ",no_quantiles)
                else:
                    print ("no true positives")

#once we find out the sensors, percentile and threshold which gives us the best result, we can print the time of attack as detected by our 
#model to see which attacks have been identified correctly from those attacks mentioned in the png file provided with the dataset
no_quantiles = 5
threshold = 0.0002
#set to hold all unique attacks identified by our model
attacks = set()
true_positive_counts_of_all_sensors = 0
no_of_alarms = 0
for fields in ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5','L_T6', 'L_T7', 'F_PU1','F_PU2', 'F_PU4', 'F_PU6', 'F_PU7', 'F_PU8', 'F_PU10','F_PU11', 'F_V2', 'S_V2', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415','P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']:
        series = read_3_series(fields)
        print ("for sensor", fields)
        _, bins = pd.qcut(series, no_quantiles, retbins=True, duplicates = 'drop')
        cut_data, bins = pd.qcut(series, no_quantiles, retbins=True, labels=range(len(bins)-1),duplicates = 'drop')
            
        #create a matrix of dimensions of no_quantiles
        occurrences_train = np.ones((no_quantiles,no_quantiles,no_quantiles))
        sum = 0
        #try for tri-grams
        for i in range(0,len(cut_data)-5): 
            a = cut_data[i]
            b = cut_data[i+1]
            c = cut_data[i+2]
            occurrences_train[a][b][c] = occurrences_train[a][b][c] + 1
                
        #convert to probabilities
        occurrences_train_prob = np.true_divide(occurrences_train, len(cut_data))
        series = read_4_series(fields)
        _, bins = pd.qcut(series, no_quantiles, retbins=True, duplicates = 'drop')
        cut_data, bins = pd.qcut(series, no_quantiles, retbins=True, labels=range(len(bins)-1), duplicates = 'drop')
    
        count = 0
        count_for_precision = 0
        precision_sum = 0
        true_positive_count = 0
        for i in range(0,len(cut_data)-5): 
            a = cut_data[i]
            b = cut_data[i+1]
            c = cut_data[i+2]
            if (occurrences_train_prob[a][b][c] < threshold):
                count = count + 1
                no_of_alarms = no_of_alarms + 1
                if attack_at_time_04(cut_data.index[i+2]) :
                    attacks.add(get_attack_number_04(cut_data.index[i+2]))
                    true_positive_count = true_positive_count + 1
                    true_positive_counts_of_all_sensors = true_positive_counts_of_all_sensors + 1
        
        if ((count > 0) and (true_positive_count > 0)) :
            print ("Using best parameters and threshold we get Precision: ",true_positive_count/count," for threshold: ", threshold," and number of percentiles: ",no_quantiles)
        else:
            print ("no true positives")    
            
            
#attacks detected as per https://github.com/DaanvanderValk/CDA2/blob/master/Data/BATADAL_dataset04_attacks.png
print ("Attacks detected ", attacks)
print ("Aggregated precision for effective sensors ", (true_positive_counts_of_all_sensors/no_of_alarms))