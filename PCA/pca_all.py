# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:43:36 2018

@author: sande
"""

#After finding the number of components from pca.py and threshold from training data in training_data03_residuals_plot.py
#We find finding the performance of pca on test dataset and identifying attacks detected by pca
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from Data.datareader import *
plt.style.use('ggplot')

features = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
     'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3', 'F_PU4', 'S_PU4',
     'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6', 'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8',
     'F_PU9', 'S_PU9', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2',
     'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302',
     'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']

df_train = read_3_df()
df_test = read_4_df()

# Only work with the selected features
df_train = df_train[features]
df_test = df_test[features]

# Standardize the features: the only the training set is used to fit the scaler
scaler = StandardScaler()
scaler.fit(df_train)
    
# Scale the training set and test set
df_train_transformed = scaler.transform(df_train)
df_test_transformed = scaler.transform(df_test)
    
# From pca.py , we dound that first 15 will explain 99% of variance, we use the last 10 for anamolous subspace
#and hence n_components is 25
pca = PCA(n_components=25)
pca.fit(df_train_transformed)
pca_model = pca.transform(df_test_transformed)

#threshold from training_data03_residuals_plot.py
threshold = 380
eigenvectors = pca.components_

#First 15 for normal behaviour and last 10 for anamolous behaviour

# Matrix P is used for normal subspace
P = np.transpose(eigenvectors[:-10])
P_T = np.transpose(P)
C = np.dot(P, P_T)

# Identity Matrix with the number of columns as given by training data  43 * 43
I = np.identity(43)

# y_residual is the projection of test data on anomalous subspace
y_residual = np.zeros((df_test_transformed.shape))

# Calculate projection of test data on anomalous subspace
for i in range(df_test_transformed.shape[0]):
    # Convert row to column vector
    y = np.transpose(df_test_transformed[i])
    y_residual[i] = np.dot(I - C, y)

#Squared prediction error for each y_residual
spe = np.zeros((df_test_transformed.shape[0]))

# flagged will be set to 1 if the spe is greater than the threshold 
flagged = np.zeros((df_test_transformed.shape[0]))
for i in range(df_test_transformed.shape[0]):
    spe[i] = np.square(np.sum(np.subtract(y_residual[i], df_test_transformed[i])))     
    if(spe[i] > threshold):
        flagged[i] = 1
            
df_test = df_test.assign(ResidualVector=spe)
tp = 0
fp = 0
tn = 0
fn = 0
attacks = set()
for i in range(df_test_transformed.shape[0]):
    if(attack_at_time_04(df_test.index[i]) and flagged[i] == 1.0):
        #In case of true positives, identify the number of attack
        attacks.add(get_attack_number_04(df_test.index[i]))
        tp = tp + 1
        
    if((attack_at_time_04(df_test.index[i]) == False) and flagged[i] == 1.0):
        fp = fp + 1
    if((attack_at_time_04(df_test.index[i]) == False) and flagged[i] == 0.0):
        tn = tn + 1    
    if(attack_at_time_04(df_test.index[i]) and flagged[i] == 0.0):
        fn = fn + 1
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print ("for threshold ",threshold,"true positive ",tp," false positive ",fp, " precision ",precision)

#attacks detected as per https://github.com/DaanvanderValk/CDA2/blob/master/Data/BATADAL_dataset04_attacks.png
print ("attacks identified as per Attacks featured in Training Data 2", attacks)
    