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
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    print ("mse value is ",mse)
                    if mse < best_score:
                        print ('Found a better score')
                        print (p,q,d)
                        best_score, best_cfg = mse, order
                        print('ARIMA%s MSE=%.3f' % (order,mse))
                except Exception as e: 
                    print(e)
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

fields = ['DATETIME',' L_T1']

series = read_csv('../Data/BATADAL_dataset04.csv', parse_dates=[0], index_col=0, squeeze=True,usecols=fields)
print(series.head())
print (type(series))
#series.plot()
#pyplot.show()

from statsmodels.tsa.arima_process import arma_generate_sample
import statsmodels.api as sm

res = sm.tsa.arma_order_select_ic(series)
print (res)
cut_data = pd.qcut(series,5,labels=["0", "1", "2","3","4"])
print (cut_data.head())
#cut_data[' L_T1'] = cut_data[' L_T1'].astype('category')
#cat_columns = cut_data.select_dtypes(['category']).columns
#cat_columns
print (type(cut_data))
#sns.countplot(y=' L_T1',data=cut_data)
#cut_data.plot()
#pyplot.show()
#print(series.head())
#print (series.keys())
#series.plot()
#pyplot.show()
#
#ARIMA_Helper.autocorrelation_plot(series, n_samples=60)
#pyplot.show()
#
p_values = [28,29,30,47,48,49]
d_values = [1,2]
q_values = [28,29,30,47,48,49]
z_values = range(0,3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)

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