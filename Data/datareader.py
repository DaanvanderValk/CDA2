# -*- coding: utf-8 -*-
"""
Created on Wed May 30 08:54:39 2018

@author: Daan
"""
import pandas as pd
from datetime import datetime

# Parser of stuff
def parser(x):
	return datetime.strptime(x, '%d/%m/%y %H')

# Read data from CSV to Panda dataframe
def read_3_df():
    return pd.read_csv('../Data/BATADAL_dataset03.csv', parse_dates=[0], date_parser=parser, index_col=0)

def read_4_df():
    return pd.read_csv('../Data/BATADAL_dataset04.csv', parse_dates=[0], date_parser=parser, index_col=0)

def read_test_df():
    return pd.read_csv('../Data/BATADAL_test_dataset.csv', parse_dates=[0], date_parser=parser, index_col=0)

# Read a particular column from CSV to Panda series
def read_3_series(feature):
    return pd.read_csv('../Data/BATADAL_dataset03.csv', parse_dates=[0], date_parser=parser, index_col=0, squeeze=True, usecols=['DATETIME', feature])

def read_4_series(feature):
    return pd.read_csv('../Data/BATADAL_dataset04.csv', parse_dates=[0], date_parser=parser, index_col=0, squeeze=True, usecols=['DATETIME', feature])

def read_test_series(feature):
    return pd.read_csv('../Data/BATADAL_test_dataset.csv', parse_dates=[0], date_parser=parser, index_col=0, squeeze=True, usecols=['DATETIME', feature])
