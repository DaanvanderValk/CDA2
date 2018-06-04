# -*- coding: utf-8 -*-
"""
Created on Fri May 18 09:52:25 2018

@author: Daan
"""

import pandas as pd

# Read data from CSV to Panda dataframe
def read_data():
    df = pd.read_csv("../Data/BATADAL_dataset04.csv")
    
    # Convert data to datetime
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], dayfirst=True)
    # Normalize attacks (map -999 to better suited value)
    df["ATT_FLAG"] = df["ATT_FLAG"].apply(lambda x: 0 if x < -900 else 130)
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
    print(df.columns.values)
#    ['DATETIME' ' L_T1' ' L_T2' ' L_T3' ' L_T4' ' L_T5' ' L_T6' ' L_T7'
# ' F_PU1' ' S_PU1' ' F_PU2' ' S_PU2' ' F_PU3' ' S_PU3' ' F_PU4' ' S_PU4'
# ' F_PU5' ' S_PU5' ' F_PU6' ' S_PU6' ' F_PU7' ' S_PU7' ' F_PU8' ' S_PU8'
# ' F_PU9' ' S_PU9' ' F_PU10' ' S_PU10' ' F_PU11' ' S_PU11' ' F_V2' ' S_V2'
# ' P_J280' ' P_J269' ' P_J300' ' P_J256' ' P_J289' ' P_J415' ' P_J302'
# ' P_J306' ' P_J307' ' P_J317' ' P_J14' ' P_J422' ' ATT_FLAG']
#    print(df.describe())
    
    figure_size = (20, 10)
    
    df.plot(x="DATETIME", figsize=figure_size)
    
    
    # ATTACK 1
    # Only select relevant features for the attack. Junctions: 302(?), 307, 317
    df_attack1 = df[["DATETIME", "L_T7", "F_PU10", "S_PU10", "F_PU11", "S_PU11", "P_J317", "L_T1", "P_J302", "P_J307", "P_J317", "ATT_FLAG"]]
    # Only select dates around the attack
    df_attack1 = select_dates(df_attack1, '2016-09-12', '2016-09-17')
    # Plot graph
    df_attack1.plot(x="DATETIME", figsize=figure_size, title="Attack 1")
    
    
    # ATTACK 2
    # Only select relevant features for the attack. Junctions: 302(?), 307, 317
    df_attack2 = df[["DATETIME", "L_T7", "F_PU10", "S_PU10", "F_PU11", "S_PU11", "P_J317", "L_T1", "P_J302", "P_J307", "P_J317", "ATT_FLAG"]]
    # Only select dates around the attack
    df_attack2 = select_dates(df_attack2, '2016-09-23', '2016-09-29')
    # Plot graph
    df_attack2.plot(x="DATETIME", figsize=figure_size, title="Attack 2")
    
    
    # ATTACK 3
    # Only select relevant features for the attack. Junctions: 280, 269
    df_attack3 = df[["DATETIME", "L_T1", "F_PU1", "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "P_J280", "P_J269", "ATT_FLAG"]]
    # Only select dates around the attack
    df_attack3 = select_dates(df_attack3, '2016-10-07', '2016-10-13')
    # Plot graph
    df_attack3.plot(x="DATETIME", figsize=figure_size, title="Attack 3")
    
    
    # ATTACK 4
    # Only select relevant features for the attack. Junctions: 280, 269
    df_attack4 = df[["DATETIME", "L_T1", "F_PU1", "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "P_J280", "P_J269", "ATT_FLAG"]]
    # Only select dates around the attack
    df_attack4 = select_dates(df_attack4, '2016-10-27', '2016-11-04')
    # Plot graph
    df_attack4.plot(x="DATETIME", figsize=figure_size, title="Attack 4")
    
    
    # ATTACK 5
    # Only select relevant features for the attack. Junctions: 289, 415
    df_attack5 = df[["DATETIME", "L_T4", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "ATT_FLAG"]]
    # Only select dates around the attack
    df_attack5 = select_dates(df_attack5, '2016-11-24', '2016-11-30')
    # Plot graph
    df_attack5.plot(x="DATETIME", figsize=figure_size, title="Attack 5")
    
    
    # ATTACK 6
    # Only select relevant features for the attack. Junctions: 289, 415
    df_attack6 = df[["DATETIME", "L_T4", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "P_J289", "P_J415", "ATT_FLAG"]]
    # Only select dates around the attack
    df_attack6 = select_dates(df_attack6, '2016-12-03', '2016-12-22')
    # Plot graph
    df_attack6.plot(x="DATETIME", figsize=figure_size, title="Attack 6")
    
    
    # ATTACK 7
    # Only select relevant features for the attack. Junctions: 289, 415
    df_attack7 = df[["DATETIME", "L_T4", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "P_J289", "P_J415", "ATT_FLAG"]]
    # Only select dates around the attack
    df_attack7 = select_dates(df_attack7, '2016-12-12', '2016-12-22')
    # Plot graph
    df_attack7.plot(x="DATETIME", figsize=figure_size, title="Attack 7")
    
    