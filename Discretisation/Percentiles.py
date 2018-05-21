# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:02:07 2018

@author: sande
"""
from pandas import read_csv
import pandas as pd

fields = ['DATETIME',' L_T1']

series = read_csv('../Data/BATADAL_dataset04.csv', parse_dates=[0], index_col=0, squeeze=True,usecols=fields)
print(series.head())
print (type(series))
#series.plot()
#pyplot.show()

#cut_data = pd.qcut(series,5,labels=["0", "1", "2","3","4"])
cut_data = pd.qcut(series,5)
print (cut_data.head())