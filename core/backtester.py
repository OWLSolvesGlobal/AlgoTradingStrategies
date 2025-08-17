from strategies.sma_crossover import SMACrossover
from core.data_loader import run_eda, save_raw_data_as_csv, save_processed_data
from core.trading_env import TradingEnvironment
from core.mt5_connector import connect_to_mt5, disconnect_from_mt5, fetch_historical_data
import MetaTrader5 as mt5
from numba import njit  # Ensure numba is imported for downstream modules

def run_bactest_live(symbols=["GBPJPY", "XAUUSD", "GBPCHF"], 
                                timeframes=[ 
                                                mt5.TIMEFRAME_H1,
                                                mt5.TIMEFRAME_M30]
                                                ):
    connect_to_mt5()
    data = fetch_historical_data(symbols=symbols, timeframes=timeframes , bars=99999)
    disconnect_from_mt5()

    run_eda(data)  # EDA and cleaning
    save_raw_data_as_csv(data) # saving unprocessed data
    strategy = SMACrossover(fast=50, slow=200)

    processed_data = strategy.generate_signals(data)
    save_processed_data(processed_data)  # saving processed data with features and signals

    env = TradingEnvironment(strategy=strategy, data_dict=processed_data)
    env.run()

    print("\n📊 Trade Log:")
    for symbol, tf_dict in env.results.items():
        for timeframe, result in tf_dict.items():
            print(f"\nSymbol: {symbol}, Timeframe: {timeframe}")
            for trade in result.get('history', []):
                print(trade)

    # Evaluate performance metrics
    env.evaluate_performance()
    env.overall_strategy_returns()