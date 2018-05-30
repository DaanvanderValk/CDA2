# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:17:06 2018

@author: Daan
"""
# Import functions from data reader:
# read_3_df(), read_4_df(), read_test_df(), read_3_series(feature), read_4_series(feature), read_test_series(feature)
import sys
sys.path.append('../')
from Data.datareader import *

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')


if __name__ == "__main__":
    feature = 'L_T1'
    figure_size = (15, 8)
    series = read_3_series(feature)
    print(series.head())
    
    p_values = q_values = range(6)
    # We do ARMA, not ARIMA, so we set d to 0
    d_values = [0]
    
    # Do a simple grid search for all combinations of values
    for p in p_values:
        for d in d_values:
            for q in q_values:
                param = (p, d, q)
                
                try:
                    mod = sm.tsa.statespace.SARIMAX(series, order=param, enforce_stationarity=False, enforce_invertibility=False)
                    results = mod.fit()
        
                    print('Results for ARIMA{}: AIC = {}'.format(param, results.aic))
                    print(results)
                except Exception as e:
                    print('ARIMA{} failed:'.format(param), e)
                    continue
        
    series.head(300).plot(figsize=figure_size)
        