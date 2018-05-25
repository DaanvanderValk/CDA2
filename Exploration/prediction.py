# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:17:06 2018

@author: Daan
"""

import pandas as pd
from matplotlib import pyplot
from datetime import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# Convert data to datetime
def parser(x):
	return datetime.strptime(x, '%d/%m/%y %H')

# Read data from CSV to Panda dataframe
def read_data():
    series = pd.read_csv("../Data/BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)['L_T1']
    return series

# Only select rows in a certain range of dates (between start and end)
# dataframe:    panda dataframe
# start, end:   strings in format '2016-12-31'
def select_dates(dataframe, start, end):
    mask = (dataframe['DATETIME'] >= start) & (dataframe['DATETIME'] <= end)
    return dataframe.loc[mask]

if __name__ == "__main__":
    series = read_data()
    
    # This is only data exploration; we will only look into one feature (L_T1) and the first 100 entries
    series = series.head(100)
    
    # Autocorrelation plot: good way to determine window size!
    # Dotted lines indicate statistically significant results
    autocorrelation_plot(series)
    pyplot.show()
    
    # Build an ARIMA model
    # fit model
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())
    
    # Use ARIMA to make predictions
    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
    	model = ARIMA(history, order=(5,1,0))
    	model_fit = model.fit(disp=0)
    	output = model_fit.forecast()
    	yhat = output[0]
    	predictions.append(yhat)
    	obs = test[t]
    	history.append(obs)
    	print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
