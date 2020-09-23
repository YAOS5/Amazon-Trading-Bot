import urllib3, shutil
from urllib.parse import urlencode
import math
import os


def create_directories(symbol):
    if not os.path.exists(f'./data/{symbol}'):
            os.makedirs(f'./data/{symbol}')

def get_data(symbol, interval, n_months):
    
    '''
    Documentation of the API: https://www.alphavantage.co/documentation/#intraday-extended
    max(n_months) = 24, because historical data in the past two years is available
    # '''

    create_directories(symbol)

    KEY = "VHY92LCH1E6NU73I"
    
    def for_month(month):
        print(f'getting data for {month}')
        slice = f'year{math.ceil(month / 12)}month{month % 12 if month % 12 != 0 else 12}'
        params = {'function': 'TIME_SERIES_INTRADAY_EXTENDED', 'symbol': symbol, 'interval': interval, 'slice': slice, 'apikey': KEY}
        url = 'https://www.alphavantage.co/query' + '?' + urlencode(params)
        c = urllib3.PoolManager()
        filename = f'./data/{symbol}/{month}.csv'
        with c.request('GET', url, preload_content=False) as res, open(filename, 'wb') as out_file:
            shutil.copyfileobj(res, out_file)

    for m in range(1, n_months + 1):
        for_month(m)

if __name__ == '__main__':
    # Make a folder called "data"
    get_data('AMZN', '1min', 24)