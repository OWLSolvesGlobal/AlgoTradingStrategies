import matplotlib.pyplot as plt

class SMACrossover:
    def __init__(self, fast: int=50, slow: int=200):
        self.fast = fast
        self.slow = slow
        
    def generate_signals(self, data_dict):
        """
        Generate buy/sell signals for multiple currency pairs and timeframes.
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
                # Signal Generation
                df['signal'] = 0
                df.loc[df['fast_sma'] > df['slow_sma'], 'signal'] = 1
                df.loc[df['fast_sma'] < df['slow_sma'], 'signal'] = -1
                df['position'] = df['signal'].shift(1).fillna(0)
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