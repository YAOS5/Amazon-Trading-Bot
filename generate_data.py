from alpha_vantage.timeseries import TimeSeries

api_key = "VHY92LCH1E6NU73I"

ts = TimeSeries(key=api_key, output_format="pandas")
data, meta_data = ts.get_intraday(symbol='MSFT', interval='1min', outputsize='full')
print(data)



