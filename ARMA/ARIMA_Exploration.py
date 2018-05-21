# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:27:14 2018

@author: sande
"""

import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas.compat import lmap
from scipy import stats
import numpy as np
import ARIMA_Helper
import warnings

fields = ['DATETIME', ' L_T1']

series = read_csv('../Data/BATADAL_dataset04.csv', parse_dates=[0], index_col=0, squeeze=True,usecols=fields)
print(series.head())
print (series.keys())
series.plot()
pyplot.show()

ARIMA_Helper.autocorrelation_plot(series, n_samples=1000)
pyplot.show()

p_values = [ 1, 2]
d_values = range(1, 3)
q_values = range(1, 3)
warnings.filterwarnings("ignore")
ARIMA_Helper.evaluate_models(series.values, p_values, d_values, q_values)

#model = ARIMA(series, order=(5,2,0))
#model_fit = model.fit(disp=0)
#print(model_fit.summary())
## plot residual errors
#residuals = DataFrame(model_fit.resid)
#residuals.plot()
#pyplot.show()
#residuals.plot(kind='kde')
#pyplot.show()
#print(residuals.describe())