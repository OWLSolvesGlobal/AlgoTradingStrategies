import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd

def connect_to_mt5():
    if not mt5.initialize():
        raise Exception("MT5 initialization failed, error code: {}".format(mt5.last_error()))
    print("Connected to MetaTrader 5")

def disconnect_from_mt5():
    mt5.shutdown()
    print("Disconnected from MetaTrader 5")

def fetch_historical_data(symbols, timeframes, bars=99999):
    """
    Fetch historical data for multiple symbols and timeframes.
    Returns a nested dictionary: {symbol: {timeframe: DataFrame}}
    """
    results = {}
    if isinstance(symbols, str):
        symbols = [symbols]
    if isinstance(timeframes, int):
        timeframes = [timeframes]

    for symbol in symbols:
        if not mt5.symbol_select(symbol, True):
            raise Exception(f"Symbol {symbol} not found or not visible in Market Watch")
        results[symbol] = {}
        for timeframe in timeframes:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                raise Exception(f"No data returned from MT5 for {symbol} at timeframe {timeframe}")
            data = pd.DataFrame(rates)
            data['time'] = pd.to_datetime(data['time'], unit='s')
            data = data.set_index('time')
            results[symbol][timeframe] = data[['open', 'high', 'low', 'close', 'tick_volume', 'spread']]
    return results