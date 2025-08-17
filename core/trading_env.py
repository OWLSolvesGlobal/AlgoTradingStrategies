import pandas as pd
import matplotlib.pyplot as plt

class TradingEnvironment:
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
                cash = self.cash
                position = 0
                history = []
                equity_curve = []
                for index, row in df.iterrows():
                    signal = row['signal']
                    price = row['close']
                    # Estimate current equity
                    if position > 0:
                        current_equity = position * price
                    else:
                        current_equity = cash
                    equity_curve.append((index, current_equity))
                    if signal == 1 and position == 0:
                        position = cash / price
                        cash = 0
                        history.append({
                            "datetime": index,
                            "action": "BUY",
                            "price": price,
                            "position_size": position,
                            "pnl": 0.0,
                            "cumulative_equity": 0.0,
                            "interpolated_equity": current_equity
                        })
                    elif signal == -1 and position > 0:
                        proceeds = position * price
                        entry = next(t for t in reversed(history) if t["action"] == "BUY")
                        pnl = proceeds - (entry["position_size"] * entry["price"])
                        cash = proceeds
                        history.append({
                            "datetime": index,
                            "action": "SELL",
                            "price": price,
                            "position_size": entry["position_size"],
                            "pnl": pnl,
                            "cumulative_equity": cash,
                            "interpolated_equity": cash
                        })
                        position = 0
                final_value = cash if position == 0 else position * df.iloc[-1]['close']
                print(f"Final portfolio value for {symbol} {timeframe}: {final_value}")
                self.plot_equity_curve(equity_curve, symbol, timeframe)
                self.dump_trades_to_csv(history, symbol, timeframe)
                self.results[symbol][timeframe] = {
                    "history": history,
                    "final_value": final_value,
                    "equity_curve": equity_curve
                }
        return self.results

    def dump_trades_to_csv(self, history, symbol, timeframe, base_path="trades"):
        trade_df = pd.DataFrame(history)
        trade_df["interpolated_equity"] = trade_df["interpolated_equity"].replace(0, method="ffill")
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
