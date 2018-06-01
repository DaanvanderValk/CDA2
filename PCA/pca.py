# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:38:53 2018

@author: Daan
"""
# Import functions from data reader:
# read_3_df(), read_4_df(), read_test_df(), read_3_series(feature), read_4_series(feature), read_test_series(feature)
import sys
sys.path.append('../')
from Data.datareader import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Consider only sensors to conduct the analysis.
    # Not considering: time, attack flag, and status of the pumps and valve (those are actuators)
    features = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
     'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3', 'F_PU4', 'S_PU4',
     'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6', 'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8',
     'F_PU9', 'S_PU9', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2',
     'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302',
     'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
    
    # Load training and test data
    df_train = read_3_df()
    df_test = read_test_df()
    
    # Load classification in a separate data frame
    df_train_flags = df_train['ATT_FLAG']
    
    # Only work with the selected features
    df_train = df_train[features]

    # Standardize the features: the only the training set is used to fit the scaler
    scaler = StandardScaler()
    scaler.fit(df_train)
    
    # Scale the training set and test set
    df_train_transformed = scaler.transform(df_train)
    df_test_transformed = scaler.transform(df_test)
    
    
    
    
    # To determine the number of components to use, try all and plot the variance that they capture
    pca = PCA()
    pca.fit(df_train_transformed)
    pca_model = pca.transform(df_train_transformed)
    x_axis = np.arange(1, len(features)+1, 1)
    plt.xlabel('Principal components')
    plt.ylabel('Variance')
    plt.axhline(y=0.99, c='darkgrey', linestyle='--', linewidth=1, label="0.99 boundary")
    plt.axhline(y=0.95, c='grey', linestyle='--', linewidth=1, label="0.95 boundary")
    plt.plot(x_axis, pca.explained_variance_ratio_.cumsum(), label="Variance captured (cumulative)")
    plt.legend(loc=4)
    plt.show()
    
#    Alternative approach:    
#    pca = PCA(.95)
#    pca.fit_transform(df_train_transformed)
#    print("PCA components for 95% of the variance:", pca.n_components_)
#    pca = PCA(.99)
#    pca.fit_transform(df_train_transformed)
#    print("PCA components for 99% of the variance:", pca.n_components_)
    
    # Results:  12 components capture 95% of the variance
    #           15 components capture 99% of the variance
    
    
    
    
    
    # Create PCA with 15 components and create a scatterplot for 2 components
    component1, component2 = 13, 14
    pca = PCA(n_components=15)
    # Fit with training set
    pca.fit(df_train_transformed)
    # Transform the training or test set
    principalComponents = pca.transform(df_train_transformed)
    
    principalComponents = principalComponents[:,[component1, component2]]

    principalDf = pd.DataFrame(data = principalComponents,
                               columns = ['principal component {}'.format(component1), 'principal component {}'.format(component2)], index=df_train.index)
    #print("principalDf:", principalDf)
    
    finalDf = pd.concat([principalDf, df_train_flags], axis = 1)
    #print("finalDf:", finalDf)
    
    # Plot the results
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component {}'.format(component1), fontsize = 15)
    ax.set_ylabel('Principal Component {}'.format(component2), fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    
    targets = [0, 1]
    colors = ['g', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['ATT_FLAG'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component {}'.format(component1)]
                   , finalDf.loc[indicesToKeep, 'principal component {}'.format(component2)]
                   , c = color
                   , s = 1)
    ax.legend(targets)
    ax.grid()
    
    print("Number of measurements attacked with 1:", len(df_train.loc[df_train_flags == 1]))
    print("Number of measurements attacked with 1:", len(finalDf.loc[finalDf['ATT_FLAG'] == 1]))
    