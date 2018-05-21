# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:02:07 2018

@author: sande
"""
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt

fields = ['DATETIME',' L_T1']

# Read a series
series = read_csv('../Data/BATADAL_dataset04.csv', parse_dates=[0], index_col=0, squeeze=True,usecols=fields)
# Also make a dataframe, for later use
df = pd.DataFrame({' L_T1':series.values}, index=series.index)

no_quantiles = 4

cut_data, bins = pd.qcut(series, no_quantiles, retbins=True, labels=range(no_quantiles))
print("Boundaries of the bins:", bins)
bin_averages = []
for i in range(no_quantiles):
    bin_averages.append((bins[i]+bins[i+1])/2)

print("Average within each bin:", bin_averages)

df['discrete'] = cut_data.apply(lambda x: bin_averages[int(x)]).astype(float)
df_part = df.head(20)

plt.figure(figsize=(9,4))
plt.plot(df_part.index, df_part[' L_T1'], marker='o', label='L_T1')
plt.step(df_part.index, df_part['discrete'], where='mid', label='L_T1 (discretized)')
plt.title('Discritization of L_T1 values into %s quantiles' % no_quantiles)

# Draw lines for bin boundaries
for i in range(1, no_quantiles):
    print("Adding line for y =", bins[i])
    plt.axhline(y=bins[i], c='grey', linestyle='--', linewidth=1, label="Quantile boundary" if i == 1 else None)
    
plt.legend(loc='upper right')