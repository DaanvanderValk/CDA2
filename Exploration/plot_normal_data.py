# -*- coding: utf-8 -*-
"""
Created on Fri May 18 09:52:25 2018

@author: Daan
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV to Panda dataframe
def read_data():
    df = pd.read_csv("../Data/BATADAL_dataset03.csv")
    
    # Convert data to datetime
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], dayfirst=True)
    return df

# Only select rows in a certain range of dates (between start and end)
# dataframe:    panda dataframe
# start, end:   strings in format '2016-12-31'
def select_dates(dataframe, start, end):
    mask = (dataframe['DATETIME'] >= start) & (dataframe['DATETIME'] <= end)
    return dataframe.loc[mask]

if __name__ == "__main__":
    df = read_data()        

#    print(df)
#    print(df.columns.values)
#    ['DATETIME' ' L_T1' ' L_T2' ' L_T3' ' L_T4' ' L_T5' ' L_T6' ' L_T7'
# ' F_PU1' ' S_PU1' ' F_PU2' ' S_PU2' ' F_PU3' ' S_PU3' ' F_PU4' ' S_PU4'
# ' F_PU5' ' S_PU5' ' F_PU6' ' S_PU6' ' F_PU7' ' S_PU7' ' F_PU8' ' S_PU8'
# ' F_PU9' ' S_PU9' ' F_PU10' ' S_PU10' ' F_PU11' ' S_PU11' ' F_V2' ' S_V2'
# ' P_J280' ' P_J269' ' P_J300' ' P_J256' ' P_J289' ' P_J415' ' P_J302'
# ' P_J306' ' P_J307' ' P_J317' ' P_J14' ' P_J422' ' ATT_FLAG']
#    print(df.describe())
    
    figure_size = (11, 6)
    #df.plot(x="DATETIME", figsize=figure_size, title='All data in one plot')
    
    df = select_dates(df, '2014-05-01', '2014-05-14')
    #df.plot(x="DATETIME", figsize=figure_size, title='All features in May 2014')
    
    df['time'] = df['DATETIME']
    df['Level of tank 1 (meters)'] = df['L_T1']
    df['Status of pump 1 (ON/OFF) x 135'] = df['S_PU1'] * 135
    df['Flow in pump 1 (LPS)'] = df['F_PU2']
    df['Status of pump 2 (ON/OFF) x 130'] = df['S_PU2'] * 130
    df['Flow in pump 2 (LPS)'] = df['F_PU2']
    df['Status of pump 3 (ON/OFF) x 125'] = df['S_PU3'] * 125
    df['Flow in pump 3 (LPS)'] = df['F_PU3']
    df['Pressure in junction 280 (meters)'] = df['P_J280']
    df['Pressure in junction 269 (meters)'] = df['P_J269']
    
    df_cluster_pu1_2_3 = df[['time', 'Level of tank 1 (meters)', 'Status of pump 1 (ON/OFF) x 135', 'Flow in pump 1 (LPS)', 'Status of pump 2 (ON/OFF) x 130', 'Flow in pump 2 (LPS)', 'Status of pump 3 (ON/OFF) x 125', 'Flow in pump 3 (LPS)', 'Pressure in junction 280 (meters)', 'Pressure in junction 269 (meters)']]
    df_cluster_pu1_2_3.plot(x="time", figsize=figure_size, title='Cluster near pump 1, 2, 3 in the first half of May 2014').legend(bbox_to_anchor=(1.00, 0.8))
    
    plt.savefig("cyclic_behavior.svg", bbox_inches='tight')
    
