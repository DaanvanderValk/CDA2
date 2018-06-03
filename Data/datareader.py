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

def attack_at_time(time):
    attacks = [
            # Dataset 4 attacks (1-7)
            [pd.to_datetime('2016-09-13 23:00:00'), pd.to_datetime('2016-09-16 00:00:00')],
            [pd.to_datetime('2016-09-26 11:00:00'), pd.to_datetime('2016-09-27 10:00:00')],
            [pd.to_datetime('2016-10-09 09:00:00'), pd.to_datetime('2016-10-11 20:00:00')],
            [pd.to_datetime('2016-10-29 19:00:00'), pd.to_datetime('2016-11-02 16:00:00')],
            [pd.to_datetime('2016-11-26 17:00:00'), pd.to_datetime('2016-11-29 04:00:00')],
            [pd.to_datetime('2016-12-06 07:00:00'), pd.to_datetime('2016-12-10 04:00:00')],
            [pd.to_datetime('2016-12-14 15:00:00'), pd.to_datetime('2016-12-19 04:00:00')],
            # Test set attacks (8-14)
            [pd.to_datetime('2017-01-16 09:00:00'), pd.to_datetime('2017-01-19 06:00:00')],
            [pd.to_datetime('2017-01-30 08:00:00'), pd.to_datetime('2017-02-02 00:00:00')],
            [pd.to_datetime('2017-02-09 03:00:00'), pd.to_datetime('2017-02-10 09:00:00')],
            [pd.to_datetime('2017-02-12 01:00:00'), pd.to_datetime('2017-02-13 07:00:00')],
            [pd.to_datetime('2017-02-24 05:00:00'), pd.to_datetime('2017-02-28 08:00:00')],
            [pd.to_datetime('2017-03-10 14:00:00'), pd.to_datetime('2017-03-13 21:00:00')],
            [pd.to_datetime('2017-03-25 20:00:00'), pd.to_datetime('2017-03-27 01:00:00')]
    ]
    for attack in attacks:
        if attack[0] <= time <= attack[1]:
            return True
    
    return False

def attack_at_time_04(time):
    attacks = [
            # Dataset 4 attacks (1-7)
            [pd.to_datetime('2016-09-13 23:00:00'), pd.to_datetime('2016-09-16 00:00:00')],
            [pd.to_datetime('2016-09-26 11:00:00'), pd.to_datetime('2016-09-27 10:00:00')],
            [pd.to_datetime('2016-10-09 09:00:00'), pd.to_datetime('2016-10-11 20:00:00')],
            [pd.to_datetime('2016-10-29 19:00:00'), pd.to_datetime('2016-11-02 16:00:00')],
            [pd.to_datetime('2016-11-26 17:00:00'), pd.to_datetime('2016-11-29 04:00:00')],
            [pd.to_datetime('2016-12-06 07:00:00'), pd.to_datetime('2016-12-10 04:00:00')],
            [pd.to_datetime('2016-12-14 15:00:00'), pd.to_datetime('2016-12-19 04:00:00')],
    ]
    for attack in attacks:
        if attack[0] <= time <= attack[1]:
            return True
    
    return False

def attack_at_time_test(time):
    attacks = [
            # Test set attacks (8-14)
            [pd.to_datetime('2017-01-16 09:00:00'), pd.to_datetime('2017-01-19 06:00:00')],
            [pd.to_datetime('2017-01-30 08:00:00'), pd.to_datetime('2017-02-02 00:00:00')],
            [pd.to_datetime('2017-02-09 03:00:00'), pd.to_datetime('2017-02-10 09:00:00')],
            [pd.to_datetime('2017-02-12 01:00:00'), pd.to_datetime('2017-02-13 07:00:00')],
            [pd.to_datetime('2017-02-24 05:00:00'), pd.to_datetime('2017-02-28 08:00:00')],
            [pd.to_datetime('2017-03-10 14:00:00'), pd.to_datetime('2017-03-13 21:00:00')],
            [pd.to_datetime('2017-03-25 20:00:00'), pd.to_datetime('2017-03-27 01:00:00')]
    ]
    for attack in attacks:
        if attack[0] <= time <= attack[1]:
            return True
    
    return False

def get_attack_number_04(time):
    attacks = [
            # Dataset 4 attacks (1-7)
            [pd.to_datetime('2016-09-13 23:00:00'), pd.to_datetime('2016-09-16 00:00:00')],
            [pd.to_datetime('2016-09-26 11:00:00'), pd.to_datetime('2016-09-27 10:00:00')],
            [pd.to_datetime('2016-10-09 09:00:00'), pd.to_datetime('2016-10-11 20:00:00')],
            [pd.to_datetime('2016-10-29 19:00:00'), pd.to_datetime('2016-11-02 16:00:00')],
            [pd.to_datetime('2016-11-26 17:00:00'), pd.to_datetime('2016-11-29 04:00:00')],
            [pd.to_datetime('2016-12-06 07:00:00'), pd.to_datetime('2016-12-10 04:00:00')],
            [pd.to_datetime('2016-12-14 15:00:00'), pd.to_datetime('2016-12-19 04:00:00')],
    ]
    if attacks[0][0] <= time <= attacks[0][1]:
        return 1
    if attacks[1][0] <= time <= attacks[1][1]:
        return 2
    if attacks[2][0] <= time <= attacks[2][1]:
        return 3
    if attacks[3][0] <= time <= attacks[3][1]:
        return 4
    if attacks[4][0] <= time <= attacks[4][1]:
        return 5
    if attacks[5][0] <= time <= attacks[5][1]:
        return 6
    if attacks[6][0] <= time <= attacks[6][1]:
        return 7
    
    
    return 0