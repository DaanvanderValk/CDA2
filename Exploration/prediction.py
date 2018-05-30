# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:17:06 2018

@author: Daan

This script predicts the next values for a certain feature, using moving average.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot
from datetime import datetime
from sklearn.metrics import mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Convert data to datetime
def parser(x):
	return datetime.strptime(x, '%d/%m/%y %H')

# Read data from CSV to Panda series
def read_data(feature):
    fields = ['DATETIME', feature]
    series = pd.read_csv("../Data/BATADAL_dataset03.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser, usecols=fields)
    return series

def select_dates(dataframe, start, end):
    mask = (dataframe.index >= start) & (dataframe.index <= end)
    return dataframe.loc[mask]

if __name__ == "__main__":
    feature= 'L_T1'
    series = read_data(feature)
    figure_size = (15, 7)
    
    # This is only data exploration; we will only look into one feature (L_T1) and the period we defined earlier
    series = select_dates(series, '2014-05-01', '2014-05-14')
    
    # Consider multiple window sizes
    window_sizes = [2, 3, 5, 10, 25, 40]
    sq_errors = {}
    predictions = {}
    
    # For each window size, make the predictions, save them, and calculate the error
    for window in window_sizes:
        sq_errors[window] = list()
        X = series.values
        history = [X[i] for i in range(window)]
        test = [X[i] for i in range(window, len(X))]
        predictions[window] = list()
        # walk forward over time steps in test
        for t in range(len(test)):
        	length = len(history)
        	yhat = np.mean([history[i] for i in range(length-window,length)])
        	obs = test[t]
        	predictions[window].append(yhat)
        	history.append(obs)
        	sq_error = np.square(obs-yhat)
        	#print('predicted=%f, expected=%f, error=%f' % (yhat, obs, sq_error))
        	sq_errors[window].append(sq_error)
        error = mean_squared_error(test, predictions[window])
        mape = mean_absolute_percentage_error(test, predictions[window])
        
        # Report findings.
        # For an explanation of MAPE, see Wikipedia:
        # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
        print('Window size:', window)
        print('Test MSE: %.3f' % error)
        print('Test MAPE: %.3f percent' % mape)
        
    # Plot predictions for each window size
    pyplot.figure(figsize=figure_size).suptitle('Moving average prediction for the water level in tank 1')
    for window in window_sizes:
        pyplot.plot(series[window:].index, predictions[window], label='Prediction (window size %d) (' % window + feature +')')
    
    #Plot original data in the same plot
    pyplot.plot(series[:].index, X, label='Original data (' + feature +')')
    pyplot.legend()
    pyplot.savefig("moving_average.svg", bbox_inches='tight')
    pyplot.show()

    # Plot the squared error for each prediction    
    pyplot.figure(figsize=figure_size).suptitle('Squared error per prediction')
    for window in window_sizes:
        pyplot.plot(series[window:].index, sq_errors[window], label='Squared error (window size %d)' % window)
    pyplot.legend()
    pyplot.savefig("squared_errors.svg", bbox_inches='tight')
    pyplot.show()