import matplotlib.pyplot as plt
import numpy as np
from numba import njit

class SMACrossover:
    def __init__(self, fast: int=50, slow: int=200):
        self.fast = fast
        self.slow = slow
        
    @staticmethod
    @njit
    def _generate_signals_numba(fast_sma, slow_sma):
        n = len(fast_sma)
        signal = np.zeros(n, dtype=np.int8)
        for i in range(n):
            if np.isnan(fast_sma[i]) or np.isnan(slow_sma[i]):
                signal[i] = 0
            elif fast_sma[i] > slow_sma[i]:
                signal[i] = 1
            elif fast_sma[i] < slow_sma[i]:
                signal[i] = -1
            else:
                signal[i] = 0
        return signal

    def generate_signals(self, data_dict):
        """
        Generate buy/sell signals for multiple currency pairs and timeframes using numba for speed.
        :param data_dict: Nested dict {symbol: {timeframe: DataFrame}}
        :return: Nested dict {symbol: {timeframe: DataFrame with signals}}
        """
        results = {}
        for symbol, tf_dict in data_dict.items():
            results[symbol] = {}
            for timeframe, df in tf_dict.items():
                df = df.copy()
                # Feature engineering
                df['fast_sma'] = df['close'].rolling(window=self.fast).mean()
                df['slow_sma'] = df['close'].rolling(window=self.slow).mean()
                df['returns'] = df['close'].pct_change()
                df['volatility_10'] = df['returns'].rolling(window=10).std()
                df['close_lag1'] = df['close'].shift(1)
                df['close_lag2'] = df['close'].shift(2)
                df['hour'] = df.index.hour
                df['dayofweek'] = df.index.dayofweek
                # Signal Generation with numba
                fast_sma = df['fast_sma'].values
                slow_sma = df['slow_sma'].values
                signal = SMACrossover._generate_signals_numba(fast_sma, slow_sma)
                df['signal'] = signal
                df['position'] = np.roll(signal, 1)
                df.loc[df.index[0], 'position'] = 0  # Safe assignment by label       
                results[symbol][timeframe] = df
        return results
    
    @staticmethod
    def plot_signals(data_dict):
        """
        Plot signals for all currency pairs and timeframes in the nested dict.
        :param data_dict: Nested dict {symbol: {timeframe: DataFrame}}
        """
        for symbol, tf_dict in data_dict.items():
            for timeframe, df in tf_dict.items():
                plt.figure(figsize=(14, 6))
                plt.plot(df['close'], label='Close')
                plt.plot(df['fast_sma'], label='SMA Fast')
                plt.plot(df['slow_sma'], label='SMA Slow')
                buys = df[df['signal'] == 1]
                sells = df[df['signal'] == -1]
                plt.plot(buys.index, df['close'].loc[buys.index], '^', markersize=10, color='g', label='Buy Signal')
                plt.plot(sells.index, df['close'].loc[sells.index], 'v', markersize=10, color='r', label='Sell Signal')
                plt.legend()
                plt.title(f"{symbol} {timeframe} - SMA Crossover")
                plt.grid(True)
                plt.show()