# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 10:22:52 2018

@author: sande
"""
#This part of code is used to plot the residuals and find outliers in training dataset and 
#find the best threshold which gives minimum number of false positives
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

df3 = read_3_df()
df3 = df3[features]
normalized_training = scaler.fit_transform(df3)
spe = np.zeros((normalized_training.shape[0]))
    
pca = decomposition.PCA(n_components=25)
pca.fit(normalized_training)
pca_model = pca.transform(normalized_training)
eigenvectors = pca.components_
P = np.transpose(eigenvectors[:-10])
P_T = np.transpose(P)
C = np.dot(P, P_T)

# Identity Matrix with the number of columns as given by training data  43 * 43
I = np.identity(43)
y_residual = np.zeros((normalized_training.shape))
for i in range(normalized_training.shape[0]):
    # Convert row to column vector
    y = np.transpose(normalized_training[i])
    y_residual[i] = np.dot(I - C, y)

#for threshold in range(270,560,30):
for threshold in range(200,600,20):
# flagged will be set to 1 if the spe is greater than the threshold 
    count = 0
    flagged = np.zeros((normalized_training.shape[0]))
    for i in range(normalized_training.shape[0]):
        spe[i] = np.square(np.sum(np.subtract(y_residual[i], normalized_training[i])))             
        if(spe[i] > threshold):
            flagged[i] = 1
            count = count + 1
            
    df3 = df3.assign(ResidualVector=spe)
    df3['ResidualVector'].plot(figsize=(15,5),label="Residuals in normal data ")
    plt.axhline(y=400, c='darkgrey', linestyle='--', linewidth=1, label="400 threshold")
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.savefig("normal_residuals.svg", bbox_inches='tight')
    print ("number of outliers or false positives ", count, "for threshold ", threshold)
    #print ("total ", normalized_training.shape[0])

