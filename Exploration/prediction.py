# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:17:06 2018

@author: Daan
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot
from datetime import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas import Series
from numpy import mean


# Convert data to datetime
def parser(x):
	return datetime.strptime(x, '%d/%m/%y %H')

# Read data from CSV to Panda series
def read_data():
    fields = ['DATETIME', 'L_T1']
    series = pd.read_csv("../Data/BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser, usecols=fields)
    return series

if __name__ == "__main__":
    series = read_data()
    figure_size = (11, 6)
    
    # This is only data exploration; we will only look into one feature (L_T1) and the first 100 entries
    series = series.head(30)
#    series.plot(figsize=figure_size)
    
#    # Tail-rolling average transform
#    rolling = series.rolling(window=3)
#    rolling_mean = rolling.mean()
#    print(rolling_mean.head(10))
#    # plot original and transformed dataset
#    series.plot(label='L_T1')
#    rolling_mean.plot(color='red', label='L_T1 (moving average with window size 3)')
#    pyplot.show()
    
    
    # TODO: try different window sizes
    window_sizes = [2, 3, 4, 5, 6]
    sq_errors = {}
    predictions = {}
    
    for window in window_sizes:
        sq_errors[window] = list()
        X = series.values
        history = [X[i] for i in range(window)]
        test = [X[i] for i in range(window, len(X))]
        predictions[window] = list()
        # walk forward over time steps in test
        for t in range(len(test)):
        	length = len(history)
        	yhat = mean([history[i] for i in range(length-window,length)])
        	obs = test[t]
        	predictions[window].append(yhat)
        	history.append(obs)
        	sq_error = np.square(obs-yhat)
        	#print('predicted=%f, expected=%f, error=%f' % (yhat, obs, sq_error))
        	sq_errors[window].append(sq_error)
        error = mean_squared_error(test, predictions[window])
        print('Window size:', window)
        print('Test MSE: %.3f' % error)
        error = np.mean(sq_errors[window])
        print('Test MSE (my computation): %.3f' % error)
        
    # plot
    
    pyplot.figure(figsize=figure_size).suptitle('Data and predictions')
    for window in window_sizes:
        pyplot.plot(series[window:].index, predictions[window], label='prediction (window size=%d)' % window)
        
    pyplot.plot(series[:].index, X, label='Original data', linewidth=3)
    pyplot.legend()
    pyplot.show()
    
    pyplot.figure(figsize=figure_size).suptitle('Squared error of prediction')
    for window in window_sizes:
        pyplot.plot(series[window:].index, sq_errors[window], label='Squared error (window size=%d)' % window)
    #pyplot.plot(series[window:].index, sq_errors[window], color='red')
    pyplot.legend()
    pyplot.show()
    
#    # Autocorrelation plot: good way to determine window size!
#    # Dotted lines indicate statistically significant results
#    autocorrelation_plot(series)
#    pyplot.show()
#    
#    # Build an ARIMA model
#    # fit model
#    model = ARIMA(series, order=(5,1,0))
#    model_fit = model.fit(disp=0)
#    print(model_fit.summary())
#    # plot residual errors
#    residuals = pd.DataFrame(model_fit.resid)
#    residuals.plot()
#    pyplot.show()
#    residuals.plot(kind='kde')
#    pyplot.show()
#    print(residuals.describe())
#    
#    # Use ARIMA to make predictions
#    X = series.values
#    size = int(len(X) * 0.66)
#    train, test = X[0:size], X[size:len(X)]
#    history = [x for x in train]
#    predictions = list()
#    for t in range(len(test)):
#    	model = ARIMA(history, order=(5,1,0))
#    	model_fit = model.fit(disp=0)
#    	output = model_fit.forecast()
#    	yhat = output[0]
#    	predictions.append(yhat)
#    	obs = test[t]
#    	history.append(obs)
#    	print('predicted=%f, expected=%f' % (yhat, obs))
#    error = mean_squared_error(test, predictions)
#    print('Test MSE: %.3f' % error)
#    # plot
#    pyplot.plot(test)
#    pyplot.plot(predictions, color='red')
#    pyplot.show()
