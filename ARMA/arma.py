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

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pprint import pprint
#plt.style.use('fivethirtyeight')

def predict(constant, err, ar_coefficients, ma_coefficients, history, history_err):
    pred = constant #+ err
    
    # Autoregressive part = done
    for i in range(1, len(ar_coefficients)+1):
        pred += ar_coefficients[i-1] * history[-i]
        print("Adding (AR):", ar_coefficients[i-1], "*", history[-i], "=", ar_coefficients[i-1] * history[-i])
        
    # Moving average part
    for i in range(1, len(ma_coefficients)+1):
        pred += ma_coefficients[i-1] * history[-i]
        print("Adding (MA):", ma_coefficients[i-1], "*", history[-i], "=", ma_coefficients[i-1] * history[-i] * history_err[-i] )
    return pred

if __name__ == "__main__":
    figure_size = (15, 8)
    
    # Features to model in ARMA: the SENSORS
    # Not considering: time, attack flag, and status of the pumps and valve (those are actuators)
#    features = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1',
#    'F_PU2', 'F_PU3', 'F_PU4', 'F_PU5', 'F_PU6', 'F_PU7', 'F_PU8', 'F_PU9', 'F_PU10',
#    'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
#    'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
    
    feature = 'L_T1'
    series_training = read_3_series(feature)
    series_test = read_test_series(feature)
    
    print("Making ARMA model")
    mod = ARMA(series_training, order=(7, 5))
    model_fit = mod.fit()

#    
#    print("Constant value:", model_fit)
    
    print("Prediction:")
    print(predict(model_fit.params[0], model_fit.bse[0], model_fit.arparams, model_fit.maparams, series_training.tail(7), model_fit.resid.tail(7)))
    
    
    
    
    
    
    print("Final 7 entries of the series:")
    print(series_training.tail(7))
    
    # one-step out-of sample forecast
    forecast = model_fit.forecast()[0]
    print("Forecast:", forecast)
    
    
    # get what you need for predicting one-step ahead
    params = model_fit.params
    residuals = model_fit.resid
    p = model_fit.k_ar
    q = model_fit.k_ma
    k_exog = model_fit.k_exog
    k_trend = model_fit.k_trend
    steps = 1
    
    print("Start of test data (including predicted value):")
    print(series_test.head(8))
    
    for steps in
    y = _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=series_test, exog=None, start=7)
    print("Prediction:", y)
    