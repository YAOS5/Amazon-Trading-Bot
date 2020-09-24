import pandas as pd
import os

def load_data(months, instrument='AMZN'):
    ''' Returns pandas dataframe for one instrument
    Inputs:
        months     - list of month ints to be loaded over the two years e.g. [11, 12, 13, 14]
        instrument - str instrument to be loaded in for those months e.g. 'GOOG'
    Outputs:
        data       - pd.DataFrame
    '''
    data = pd.DataFrame()
    
    for month in months:
        # Sorry, only works for Windows atm, please update so it works for both and remove this comment
        monthly_data = pd.read_csv(os.getcwd() + f'\data\\{instrument}\\{month}.csv')
        # Assuming we only want to keep time, close and volume - removing open, high and low
        data = pd.concat([data, monthly_data[['time', 'close', 'volume']]], axis=0)
        
    # Sort by time and reset index
    data['time'] = pd.to_datetime(data.time)
    return data.sort_values('time').reset_index().drop('index', axis=1)