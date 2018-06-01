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

# See: http://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARMA.html
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
from scipy import stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot
from pprint import pprint
# ARMA typically generates many errors: ignore these
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    figure_size = (9, 3.5)
    
    # Features to model in ARMA: the SENSORS
    # Not considering: time, attack flag, and status of the pumps and valve (those are actuators)
#    features = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1',
#    'F_PU2', 'F_PU3', 'F_PU4', 'F_PU5', 'F_PU6', 'F_PU7', 'F_PU8', 'F_PU9', 'F_PU10',
#    'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
#    'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
    
    feature = 'L_T1'
    order = (7, 5)
    measures_needed = np.max(order) # Number of measures needed: maximum of p and q
    series_training = read_3_series(feature)
    series_test = read_test_series(feature)
    test_length = len(series_test)
    
    # Generate the ARMA model, using only the training data
    mod = ARMA(series_training, order=(7, 5))
    model_fit = mod.fit()
    
#    # Print relevant details of the generated model
#    print(model_fit.summary())
    
    # Get the parameters from the ARMA model to make out-of-sample predictions
    params = model_fit.params
    p = model_fit.k_ar
    q = model_fit.k_ma
    k_exog = model_fit.k_exog
    k_trend = model_fit.k_trend
    
    # Determine the 80-percentile
    training_residuals = np.absolute(model_fit.resid)
    print("Residuals:", training_residuals.describe())
    threshold = np.percentile(training_residuals, 95)
    print("95-percentile:", threshold)
    
    # Store the predictions and their residual errors in arrays
    predictions = np.zeros(test_length)
    prediction_residuals = np.zeros(test_length)
        # (Note that the residuals are initially set 0, because you cannot get a prediction for the first values
        #  as you need a couple of measurements first. This means that the first predictions can be off a bit.)
    prediction_residuals_abs = np.zeros(test_length)
    count = 0
    count_pos = 0
    #threshold = 0.3
    
    # Make the predictions for the test data, using only the ARMA model generated with the training data
    for position in range(measures_needed, test_length):
        #print("Looking at residuals:", prediction_residuals[position-measures_needed:position])
        #print("And previous values:", series_test[position-measures_needed:position])
        predictions[position] = _arma_predict_out_of_sample(params, 1, prediction_residuals[position-measures_needed:position], p, q, k_trend, k_exog, endog=series_test[position-measures_needed:position], exog=None, start=measures_needed)
        prediction_residuals[position] = series_test[position] - predictions[position]
        resi_abs = np.abs(series_test[position] - predictions[position])
        prediction_residuals_abs[position] = resi_abs
        # We don't throw any alarms for the first max(p, q) predictions,
        # as the first couple of predictions are typically more off then others
        if position > 2 * measures_needed and resi_abs > threshold:
            count += 1
            time = series_test[position:position+1].index
            #print("Alert on {}: {} positive".format(time, attack_at_time(time)))
            if(attack_at_time(time)):
                count_pos += 1
            
        
    
    tpr = count_pos / count * 100
    print("RAISED ALARMS: True positive rate: {} ({}/{})".format(tpr, count_pos, count))
        
#    print("Real values")
#    print(series_test[7:15])
#    print("Predictions")
#    print(predictions[7:15])
#    print("Residuals")
#    print(prediction_residuals[7:15])
    
    pyplot.figure(figsize=figure_size)
    
    start = measures_needed
    end = 100 + measures_needed #maximum: test_length
    
    pyplot.plot(series_test[start:end].index, predictions[start:end], label='Predictions')
    pyplot.plot(series_test.index[start:end], series_test[start:end], label='Real values (test set)')
    pyplot.plot(series_test.index[start:end], prediction_residuals[start:end], color='red', label='Residuals (errors)')
    
    pyplot.legend(loc=1)
    pyplot.savefig("armapredictions.svg", bbox_inches='tight')
    pyplot.show()
    
    print("Description of the residuals:")
    print(stats.describe(prediction_residuals[measures_needed:]))
    print("Description of the residuals (abs):")
    print(stats.describe(prediction_residuals_abs[measures_needed:]))