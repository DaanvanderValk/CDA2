# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:34:59 2018

@author: Daan
"""
# Import functions from data reader:
# read_3_df(), read_4_df(), read_test_df(), read_3_series(feature), read_4_series(feature), read_test_series(feature)
import sys
sys.path.append('../')
from Data.datareader import *

df = read_3_series('L_T1')
print(df)