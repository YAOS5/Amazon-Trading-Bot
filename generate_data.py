# from alpha_vantage.timeseries import TimeSeries

# function="TIME_SERIES_INTRADAY_EXTENDED"
# api_key = "VHY92LCH1E6NU73I"

# ts = TimeSeries(key=api_key, output_format="pandas")
# data, meta_data = ts.get_intraday(symbol='MSFT', interval='1min', outputsize='full', slice="year1month12")
# print(data)


import urllib3, shutil
url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=15min&slice=year1month1&apikey=demo"
c = urllib3.PoolManager()
filename = "test.csv"
with c.request('GET', url, preload_content=False) as res, open(filename, 'wb') as out_file:
    shutil.copyfileobj(res, out_file)