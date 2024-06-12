import requests
import pandas as pd

class CoinGecko:
  def __init__(self):
    pass

  def get_price(self, token_name, _from: str, _to: str):
    url = f'https://api.coingecko.com/api/v3/coins/{token_name}/market_chart'

    start_date = pd.to_datetime(_from)
    end_date = pd.to_datetime(_to)
    days_back = (end_date - start_date).days
    
    start_timestamp = int(start_date.timestamp()) * 1000
    end_timestamp = int(end_date.timestamp()) * 1000

    params = {
        'vs_currency': 'usd',
        'days': days_back,
        'from': start_timestamp,
        'to': end_timestamp
    }

    # Make the API request
    response = requests.get(url, params=params)
    
    data = response.json()

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
