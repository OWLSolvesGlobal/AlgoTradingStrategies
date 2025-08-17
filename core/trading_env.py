import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

class TradingEnvironment:
    @staticmethod
    @njit
    def _fast_backtest(signals, prices, initial_cash):
        n = len(signals)
        cash = initial_cash
        position = 0.0
        equity_curve = np.zeros(n)
        for i in range(n):
            signal = signals[i]
            price = prices[i]
            if position > 0:
                current_equity = position * price
            else:
                current_equity = cash
            equity_curve[i] = current_equity
            if signal == 1 and position == 0:
                position = cash / price
                cash = 0
            elif signal == -1 and position > 0:
                proceeds = position * price
                cash = proceeds
                position = 0
        final_value = cash if position == 0 else position * prices[-1]
        return equity_curve, final_value
    def overall_strategy_returns(self):
        """
        Aggregate and display overall strategy returns across all symbols and timeframes.
        Returns a summary DataFrame for further analysis or visualization.
        """
        summary = []
        for symbol, tf_dict in self.results.items():
            for timeframe, result in tf_dict.items():
                equity_curve = result["equity_curve"]
                if not equity_curve:
                    continue
                equity_series = pd.Series([val for _, val in equity_curve])
                returns = equity_series.pct_change().dropna()
                total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
                max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
                sharpe = (returns.mean() / returns.std()) * (252 * 24 * 4) ** 0.5 if returns.std() != 0 else float('nan')
                summary.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "sharpe": sharpe,
                    "final_value": result["final_value"]
                })
        summary_df = pd.DataFrame(summary)
        print("\n=== Overall Strategy Returns Summary ===")
        print(summary_df)
        # Optionally plot aggregate returns
        if not summary_df.empty:
            plt.figure(figsize=(10, 5))
            plt.bar(summary_df["symbol"] + "_" + summary_df["timeframe"].astype(str), summary_df["total_return"])
            plt.title("Total Return by Symbol/Timeframe")
            plt.ylabel("Total Return")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()
        return summary_df
    def __init__(self, strategy, data_dict, cash=1000):
        self.strategy = strategy
        self.data_dict = data_dict  # Nested dict {symbol: {timeframe: DataFrame}}
        self.cash = cash
        self.results = {}  # Store results for each symbol/timeframe

    def run(self):
        # Generate signals for all pairs/timeframes
        signal_dict = self.strategy.generate_signals(self.data_dict)
        for symbol, tf_dict in signal_dict.items():
            self.results[symbol] = {}
            for timeframe, df in tf_dict.items():
                signals = df['signal'].values.astype(np.int8)
                prices = df['close'].values.astype(np.float64)
                equity_curve, final_value = self._fast_backtest(signals, prices, self.cash)
                # Reconstruct equity_curve with timestamps for plotting and saving
                equity_curve_list = list(zip(df.index, equity_curve))
                print(f"Final portfolio value for {symbol} {timeframe}: {final_value}")
                self.plot_equity_curve(equity_curve_list, symbol, timeframe)
                # For simplicity, skip trade history in numba version
                self.dump_trades_to_csv(
                    [{'datetime': idx, 'interpolated_equity': eq} for idx, eq in equity_curve_list],
                    symbol, timeframe)
                self.results[symbol][timeframe] = {
                    "history": [],
                    "final_value": final_value,
                    "equity_curve": equity_curve_list
                }
        return self.results

    def dump_trades_to_csv(self, history, symbol, timeframe, base_path="trades"):
        trade_df = pd.DataFrame(history)
        trade_df["interpolated_equity"] = trade_df["interpolated_equity"].replace(0, np.nan).ffill()
        folder = f"{base_path}/{symbol}"
        import os
        os.makedirs(folder, exist_ok=True)
        path = f"{folder}/{timeframe}_trades.csv"
        trade_df.to_csv(path, index=False)
        print(f"📁 Trade log saved to {path}")

    def plot_equity_curve(self, equity_curve, symbol, timeframe):
        if not equity_curve:
            print(f"⚠️ No equity curve available to plot for {symbol} {timeframe}.")
            return
        dates, equity_values = zip(*equity_curve)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity_values, label="Interpolated Equity")
        plt.title(f"Portfolio Equity Curve: {symbol} {timeframe}")
        plt.xlabel("Time")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def evaluate_performance(self):
        metrics = []
        for symbol, tf_dict in self.results.items():
            for timeframe, result in tf_dict.items():
                equity_curve = result["equity_curve"]
                if not equity_curve:
                    print(f"No equity curve to evaluate for {symbol} {timeframe}.")
                    continue
                equity_series = pd.Series([val for _, val in equity_curve])
                returns = equity_series.pct_change().dropna()
                total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
                max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
                sharpe = (returns.mean() / returns.std()) * (252 * 24 * 4) ** 0.5 if returns.std() != 0 else float('nan')
                print(f"Performance for {symbol} {timeframe}:")
                print(f"  Total Return: {total_return:.2%}")
                print(f"  Max Drawdown: {max_drawdown:.2%}")
                print(f"  Sharpe Ratio: {sharpe:.2f}")
                metrics.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "sharpe": sharpe,
                    "final_value": result["final_value"]
                })
        return pd.DataFrame(metrics)
