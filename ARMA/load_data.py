# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:34:59 2018

@author: Daan
"""

import sys
sys.path.append('../')
from Data.datareader import read_3_df
from Data.datareader import read_4_df
from Data.datareader import read_test_df
from Data.datareader import read_3_series
from Data.datareader import read_4_series
from Data.datareader import read_test_series

df = read_3_df()
print(df)