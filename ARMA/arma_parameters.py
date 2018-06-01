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

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from pandas.compat import lmap

def autocorrelation_plot_limited(series, n_samples=None, ax=None, **kwds):
    """Autocorrelation plot for time series.
    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    kwds : keywords
        Options to pass to matplotlib plotting method
    Returns:
    -----------
    ax: Matplotlib axis object
    """
    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca(xlim=(1, n_samples), ylim=(-1.0, 1.0))
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) *
                (data[h:] - mean)).sum() / float(n) / c0
    x = (np.arange(n) + 1).astype(int)
    y = lmap(r, x)
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    # Only show the 99% confidence interval
    ax.axhline(y=z99 / np.sqrt(n), linestyle='--', color='grey')
#    ax.axhline(y=z95 / np.sqrt(n), color='grey')
    ax.axhline(y=0.0, color='black')
#    ax.axhline(y=-z95 / np.sqrt(n), color='grey')
    ax.axhline(y=-z99 / np.sqrt(n), linestyle='--', color='grey')
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    if n_samples:
        ax.plot(x[:n_samples], y[:n_samples], **kwds)
    else:
        ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    return ax

if __name__ == "__main__":
    figure_size = (4, 2.5)
    
    # Features to model in ARMA: the SENSORS
    # Not considering: time, attack flag, and status of the pumps and valve (those are actuators)
    features = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1',
    'F_PU2', 'F_PU3', 'F_PU4', 'F_PU5', 'F_PU6', 'F_PU7', 'F_PU8', 'F_PU9', 'F_PU10',
    'F_PU11', 'F_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
    'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
    
    # We try to find the best ARMA order (p, q) to model each sensor
    
    
    # APPROACH 1: Plot and study the autocorrelation plots for each sensor
    feature = 'L_T1'
    series = read_3_series(feature)
    plt.figure(figsize=figure_size).suptitle('Autocorrelation')
    autocorrelation_plot_limited(series, n_samples=100, label='{}'.format(feature))
    plt.savefig('autocorrelation.svg', bbox_inches='tight')
    plt.show()
    
    
    
    
    # APPROACH 2: Try different ARMA parameters and calculate AIC scores to find the best fit per sensor
    
    # Prepare a simple grid search for ARMA parameters p and q.
    # Default values (in the Python implementation) are p, q = (2, 2).
    # We found that values for p and q larger than 7 typically result
    # in crashes and exceptions when building the model.
    # So, we do a grid search for range [0..7] for both p and q.
    p_values = q_values = range(0, 8)
    
    # To score the predictions of a particular model, we measure the
    # Akaike information criterion (AIC), which should be as low as possible.
    # See: https://en.wikipedia.org/wiki/Akaike_information_criterion
    
    print('Use a grid search for the best ARMA parameter for each sensor in the dataset.')
    for feature in features:
        series = read_3_series(feature)
        
        # We want to select the minimal AIC, so keep track of the used parameters and best score.
        aic_minimum = np.inf
        aic_minimum_params = (np.nan, np.nan)
        for p in p_values:
                for q in q_values:
                    param = (p, q)
                    
                    try:
                        mod = ARMA(series, order=param)
                        results = mod.fit()
                        
                        aic = results.aic
                        if aic < aic_minimum:
                            aic_minimum = aic
                            aic_minimum_params = param
            
                        #print('Results for ARMA{}: AIC = {}'.format(param, aic))
                        
                    # When building the ARMA models, anticipate exceptions!
                    except Exception as e:
                        #print('Results for ARMA{}: failed due to exception'.format(param))
                        continue
        
        # Report on the best parameters for this feature.
        print('Best ARMA parameters (p, q) for feature {}: {}'.format(feature, aic_minimum_params), 'yielding an AIC of {}'.format(aic_minimum))
        