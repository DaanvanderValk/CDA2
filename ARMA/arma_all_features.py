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
import numpy as np
# ARMA typically generates many errors: ignore these
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    figure_size = (9, 3.5)
    
    # Features to model in ARMA: the SENSORS
    # Not considering: time, attack flag, and status of the pumps and valve (those are actuators)
    # These data is obtained from executing arma_parameters.py;
    # the results are copied from arma_parameter_results.txt
    
    # Percentile of (absolute) errors to be taken as threshold
    percentile_threshold = 95
    
    # Pumps 3, 5 and 9 could not be modelled.
    
    arma_order_per_feature = {
        'L_T1': (7, 5), # AIC: -13649.29592566039
        'L_T2': (4, 6), # AIC: -1895.4959046989898
        'L_T3': (7, 7), # AIC: -9375.915544908556
        'L_T4': (3, 7), # AIC: 7886.873356320462
        'L_T5': (6, 7), # AIC: -829.1282689760119
        'L_T6': (6, 5), # AIC: -12828.443875078872
        'L_T7': (4, 7), # AIC: 13983.570768549054
        'F_PU1': (7, 6), # AIC: 50828.50995193154
        'F_PU2': (4, 6), # AIC: 75135.71891938259
#        'F_PU3': (nan, nan), # AIC: inf
        'F_PU4': (5, 7), # AIC: 63119.37773569395
#        'F_PU5': (nan, nan), # AIC: inf
        'F_PU6': (1, 2), # AIC: 28585.854927263095
        'F_PU7': (5, 6), # AIC: 73514.75746342457
        'F_PU8': (6, 5), # AIC: 66589.57603864257
#        'F_PU9': (nan, nan), # AIC: inf
        'F_PU10': (4, 5), # AIC: 65917.53775714341
        'F_PU11': (0, 1), # AIC: 7079.514595597204
        'F_V2': (7, 6), # AIC: 77905.68138735935
        'P_J280': (7, 7), # AIC: -78870.98496242735
        'P_J269': (7, 6), # AIC: 40411.01308753596
        'P_J300': (4, 6), # AIC: 31467.665911501295
        'P_J256': (7, 7), # AIC: 50285.35384538233
        'P_J289': (7, 7), # AIC: 31473.343983911396
        'P_J415': (7, 7), # AIC: 59207.211159791696
        'P_J302': (7, 6), # AIC: 44563.12859184672
        'P_J306': (5, 7), # AIC: 57111.21100354387
        'P_J307': (7, 6), # AIC: 44688.812368328254
        'P_J317': (7, 4), # AIC: 52235.39952149622
        'P_J14': (7, 6), # AIC: 40794.59488141292
        'P_J422': (5, 7), # AIC: 30717.203970890823
    }
    
    found_attacks_arma = set()
    
    for feature in arma_order_per_feature.keys():
        order = arma_order_per_feature[feature]
        measures_needed = np.max(order) # Number of measures needed: maximum of p and q
        series_training = read_3_series(feature)
        series_test = read_test_series(feature)
        test_length = len(series_test)
        
        print()
        print("Generating and evaluating ARMA model of order {} for sensor {}:".format(feature, order))
        
        # Generate the ARMA model, using only the training data
        mod = ARMA(series_training, order=order)
        model_fit = mod.fit()
        
        # Get the parameters from the ARMA model to make out-of-sample predictions
        params = model_fit.params
        p = model_fit.k_ar
        q = model_fit.k_ma
        k_exog = model_fit.k_exog
        k_trend = model_fit.k_trend
        
        # Print model quality measures (of the predictions for the training data)
        aic = model_fit.aic
        print("Model performance: AIC = {}".format(aic))
        
        # Determine the 95-percentile of the absolute value of the residuals in the training set predictions
        training_residuals_abs = np.absolute(model_fit.resid)
        threshold = np.percentile(training_residuals_abs, percentile_threshold)
        print("Used threshold ({}-percentile): {}".format(percentile_threshold, threshold))
        
        # Store the predictions and their residual errors in arrays
        predictions = np.zeros(test_length)
        prediction_residuals = np.zeros(test_length)
            # (Note that the residuals are initially set 0, because you cannot get a prediction for the first values
            #  as you need a couple of measurements first. This means that the first predictions can be off a bit.)
        prediction_residuals_abs = np.zeros(test_length)
        count = 0
        count_pos = 0
        found_attacks = set()
        
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
                # Raise alarm!
                count += 1
                time = series_test[position:position+1].index
                attack_no = get_attack_number(time)
                
                if(attack_no != 0):
                    count_pos += 1
                    found_attacks.add(attack_no)
                
            
        
        precision = count_pos / count * 100
        print("Precision: {}% ({}/{}).".format(precision, count_pos, count))
        print("Identified attacks:", found_attacks)
        
        # Update overall attacks identified by ARMA
        found_attacks_arma.update(found_attacks)