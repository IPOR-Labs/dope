import requests
import pandas as pd

class CoinGecko:
  def __init__(self):
    pass
  
  def _parse_price(self, data):
    # Extract the prices (timestamp and closing price)
    if 'prices' not in data:
      raise RuntimeError(data)
    prices = data['prices']

    # Convert the data to a DataFrame
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])

    # Convert timestamp to readable date
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set the date as the index
    df.set_index('date', inplace=True)
    df.drop(columns=['timestamp'], inplace=True)

    return df
    
  
  def get_last_year_price(self, token_name,):
    # https://docs.coingecko.com/v3.0.1/reference/coins-id-market-chart
    url = f'https://api.coingecko.com/api/v3/coins/{token_name}/market_chart'

    params = {
        'vs_currency': 'usd',
        'days': 365,
    }

    # Make the API request
    response = requests.get(url, params=params)
    
    data = response.json()
    df = self._parse_price(data)
    return df.resample("1D").last()
    

  def get_price(self, token_name, _from: str, _to: str):
    # https://docs.coingecko.com/v3.0.1/reference/coins-id-market-chart-range
    url = f'https://api.coingecko.com/api/v3/coins/{token_name}/market_chart/range'

    start_date = pd.to_datetime(_from)
    end_date = pd.to_datetime(_to)
    
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    params = {
        'vs_currency': 'usd',
        'from': start_timestamp,
        'to': end_timestamp
    }

    # Make the API request
    response = requests.get(url, params=params)
    
    data = response.json()
    df = self._parse_price(data)
    return df.resample("1D").last()
