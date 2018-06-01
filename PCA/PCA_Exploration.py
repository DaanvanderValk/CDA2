# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:00:37 2018

@author: sande
"""

from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

fields = ['DATETIME',' L_T1']

# Read a series
#series = read_csv('../Data/BATADAL_dataset04.csv', parse_dates=[0], index_col=0, squeeze=True,usecols=fields)
series = read_csv('../Data/BATADAL_dataset04.csv', parse_dates=[0], index_col=0, squeeze=True)
pca = PCA()
pca.fit(series)
print (pca.components_)