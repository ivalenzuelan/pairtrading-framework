#!/usr/bin/env python3
"""
Comprehensive Cryptocurrency Pair Trading Backtester
Integrates pair selection, strategy parameterization, and backtesting
Supports: Cointegration, Distance, Correlation, SDR, and Hurst metrics
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tools.sm_exceptions import CollinearityWarning
from hurst import compute_Hc
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore', category=CollinearityWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
def create_default_config():
    """Create default configuration dictionary."""
    return {
        "cryptos": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"],
        "intervals": ["1h"],
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "metrics": ["cointegration", "distance", "correlation"],
        "selection": {
            "top_n": 5,
            "cointegration_threshold": 0.05,
            "distance_threshold": 0.8,
            "correlation_threshold": 0.7,
            "sdr_threshold": 0.5,
            "hurst_threshold": 0.4
        },
        "strategy": {
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_loss": 3.0,
            "take_profit": 2.0,
            "lookback_window": 24 * 7,  # 1 week for hourly data
            "max_trade_duration": 72,    # 72 hours
            "position_size": 1000,       # $1000 per leg
            "trading_fee": 0.001,       # 0.1% per trade
            "slippage": 0.0005          # 0.05%
        },
        "walkforward": {
            "train_days": 30,
            "trade_days": 5,
            "buffer_days": 3
        },
        "output_dir": "backtest_results"
    }

# ----------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# ----------------------------------------------------------------------
def load_crypto_data(file_path, symbol):
    """Load cryptocurrency data from CSV or Parquet file."""
    # Simplified implementation - replace with your actual data loading logic
    # This would typically load OHLCV data and return a DataFrame with datetime index
    # For demo purposes, we'll generate random data
    dates = pd.date_range(start="2023-01-01", end="2023-06-30", freq="1h")
    prices = np.exp(np.cumsum(np.random.normal(0, 0.001, len(dates))))
    return pd.DataFrame({symbol: prices}, index=dates)

def merge_data(data_dict, interval):
    """Merge data from multiple symbols into a single DataFrame."""
    # Align all series on a common index
    merged = pd.concat([df for df in data_dict.values()], axis=1)
    merged.columns = list(data_dict.keys())
    return merged.dropna()

# ----------------------------------------------------------------------
# STATISTICAL METRICS CALCULATION
# ----------------------------------------------------------------------
def calculate_cointegration(s1, s2):
    """Calculate cointegration metrics between two series."""
    try:
        # Use log prices for cointegration test
        s1_log, s2_log = np.log(s1), np.log(s2)
        coint_res = coint(s1_log, s2_log, autolag='BIC')
        
        # Johansen test
        df = pd.DataFrame({'s1': s1_log, 's2': s2_log}).dropna()
        johansen_res = coint_johansen(df, det_order=-1, k_ar_diff=1)
        
        return {
            'eg_pvalue': coint_res[1],
            'johansen_trace': johansen_res.lr1[0],
            'johansen_crit': johansen_res.cvt[0, 1]  # 95% critical value
        }
    except:
        return {'eg_pvalue': 1.0, 'johansen_trace': 0, 'johansen_crit': 0}

def calculate_distance(s1, s2):
    """Calculate normalized distance metric between two series."""
    try:
        # Normalize log prices
        s1_log, s2_log = np.log(s1), np.log(s2)
        s1_norm = (s1_log - s1_log.mean()) / s1_log.std()
        s2_norm = (s2_log - s2_log.mean()) / s2_log.std()
        return np.mean(np.abs(s1_norm - s2_norm))
    except:
        return np.nan

def calculate_correlation(s1, s2):
    """Calculate Pearson correlation of returns."""
    try:
        ret1 = np.log(s1).diff().dropna()
        ret2 = np.log(s2).diff().dropna()
        common_idx = ret1.index.intersection(ret2.index)
        return ret1.loc[common_idx].corr(ret2.loc[common_idx])
    except:
        return 0.0

def calculate_sdr(s1, s2):
    """Calculate Signal-to-Drift Ratio."""
    try:
        ratio = s1 / s2
        returns = np.log(ratio).diff().dropna()
        return np.mean(returns) / np.std(returns)
    except:
        return 0.0

def calculate_hurst(series):
    """Calculate Hurst exponent for mean-reversion detection."""
    try:
        H, c, data = compute_Hc(series, kind='price', simplified=True)
        return H
    except:
        return 0.5  # Neutral value

# ----------------------------------------------------------------------
# PAIR SELECTION
# ----------------------------------------------------------------------
def evaluate_pair(s1, s2, metrics):
    """Evaluate a pair using all requested metrics."""
    results = {}
    
    if 'cointegration' in metrics:
        coint_metrics = calculate_cointegration(s1, s2)
        results.update(coint_metrics)
        results['cointegrated'] = (
            coint_metrics['eg_pvalue'] < 0.05 or 
            coint_metrics['johansen_trace'] > coint_metrics['johansen_crit']
        )
    
    if 'distance' in metrics:
        results['distance'] = calculate_distance(s1, s2)
    
    if 'correlation' in metrics:
        results['correlation'] = calculate_correlation(s1, s2)
    
    if 'sdr' in metrics:
        results['sdr'] = calculate_sdr(s1, s2)
    
    if 'hurst' in metrics:
        ratio = s1 / s2
        results['hurst'] = calculate_hurst(ratio)
    
    return results

def select_pairs(data, metrics, selection_criteria):
    """Select top pairs based on statistical metrics."""
    pairs = []
    symbols = data.columns.tolist()
    
    for i, sym1 in enumerate(symbols):
        for j, sym2 in enumerate(symbols[i+1:], i+1):
            s1 = data[sym1]
            s2 = data[sym2]
            
            metrics = evaluate_pair(s1, s2, metrics)
            metrics.update({
                'symbol1': sym1,
                'symbol2': sym2
            })
            
            # Apply selection thresholds
            valid = True
            if 'cointegration' in metrics and 'cointegrated' in metrics:
                valid = valid and metrics['cointegrated']
            
            if 'distance' in metrics:
                valid = valid and (metrics['distance'] < selection_criteria['distance_threshold'])
            
            if 'correlation' in metrics:
                valid = valid and (abs(metrics['correlation']) > selection_criteria['correlation_threshold'])
            
            if 'sdr' in metrics:
                valid = valid and (abs(metrics['sdr']) > selection_criteria['sdr_threshold'])
            
            if 'hurst' in metrics:
                valid = valid and (metrics['hurst'] < selection_criteria['hurst_threshold'])
            
            if valid:
                pairs.append(metrics)
    
    # Rank and select top pairs
    if 'cointegration' in metrics:
        pairs.sort(key=lambda x: x['eg_pvalue'])
    elif 'distance' in metrics:
        pairs.sort(key=lambda x: x['distance'])
    elif 'correlation' in metrics:
        pairs.sort(key=lambda x: -abs(x['correlation']))
    
    return pairs[:selection_criteria['top_n']]

# ----------------------------------------------------------------------
# STRATEGY PARAMETERIZATION
# ----------------------------------------------------------------------
def calculate_hedge_ratio(s1, s2):
    """Calculate hedge ratio using OLS regression."""
    s1_log = np.log(s1)
    s2_log = np.log(s2)
    slope, intercept, r_value, p_value, std_err = linregress(s2_log, s1_log)
    return slope

def calculate_bollinger_bands(spread, lookback):
    """Calculate Bollinger Bands parameters."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return mean, std

def prepare_pair_strategy(data, pair, strategy_params):
    """Prepare trading parameters for a pair."""
    s1 = data[pair['symbol1']]
    s2 = data[pair['symbol2']]
    
    # Calculate hedge ratio
    hedge_ratio = calculate_hedge_ratio(s1, s2)
    
    # Calculate spread
    spread = np.log(s1) - hedge_ratio * np.log(s2)
    
    # Calculate Bollinger Bands
    lookback = strategy_params['lookback_window']
    spread_mean, spread_std = calculate_bollinger_bands(spread, lookback)
    
    # Calculate z-score
    zscore = (spread - spread_mean) / spread_std
    
    return {
        'symbol1': pair['symbol1'],
        'symbol2': pair['symbol2'],
        'hedge_ratio': hedge_ratio,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'zscore': zscore
    }

# ----------------------------------------------------------------------
# TRADING EXECUTION
# ----------------------------------------------------------------------
class PairTrade:
    """Track an open pair trade."""
    def __init__(self, entry_time, symbol1, symbol2, 
                 price1, price2, hedge_ratio, position_type):
        self.entry_time = entry_time
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.entry_price1 = price1
        self.entry_price2 = price2
        self.hedge_ratio = hedge_ratio
        self.position_type = position_type  # "long" or "short"
        self.exit_time = None
        self.exit_price1 = None
        self.exit_price2 = None
        self.status = "open"  # "open", "closed", "stopped"
        
    def calculate_pnl(self, fee_rate, slippage):
        """Calculate P&L for the trade."""
        # Calculate price changes with slippage
        if self.position_type == "long":
            price1_change = (self.exit_price1 / self.entry_price1) * (1 - slippage)
            price2_change = (self.entry_price2 / self.exit_price2) * (1 - slippage)
        else:  # short
            price1_change = (self.entry_price1 / self.exit_price1) * (1 - slippage)
            price2_change = (self.exit_price2 / self.entry_price2) * (1 - slippage)
        
        # Calculate returns
        return1 = price1_change - 1
        return2 = price2_change - 1
        
        # Apply fees (entry and exit)
        net_return = return1 - (return2 * self.hedge_ratio) - (4 * fee_rate)
        return net_return

def execute_trading(data, pair_params, strategy_params):
    """Execute trading strategy for a pair."""
    trades = []
    current_trade = None
    zscore = pair_params['zscore']
    
    for timestamp, row in data.iterrows():
        # Get current prices
        price1 = row[pair_params['symbol1']]
        price2 = row[pair_params['symbol2']]
        current_z = zscore.loc[timestamp] if timestamp in zscore.index else np.nan
        
        # Skip if we don't have a valid z-score
        if np.isnan(current_z):
            continue
        
        # Check if we should open a new trade
        if current_trade is None:
            if current_z < -strategy_params['entry_z']:
                # Open long position (spread is undervalued)
                current_trade = PairTrade(
                    timestamp, pair_params['symbol1'], pair_params['symbol2'],
                    price1, price2, pair_params['hedge_ratio'], "long"
                )
            elif current_z > strategy_params['entry_z']:
                # Open short position (spread is overvalued)
                current_trade = PairTrade(
                    timestamp, pair_params['symbol1'], pair_params['symbol2'],
                    price1, price2, pair_params['hedge_ratio'], "short"
                )
        
        # Manage open trade
        elif current_trade is not None:
            # Check exit conditions
            exit_condition = False
            
            # Normal exit conditions
            if current_trade.position_type == "long" and current_z >= -strategy_params['exit_z']:
                exit_condition = True
            elif current_trade.position_type == "short" and current_z <= strategy_params['exit_z']:
                exit_condition = True
                
            # Stop loss conditions
            if current_trade.position_type == "long" and current_z <= -strategy_params['stop_loss']:
                current_trade.status = "stopped"
                exit_condition = True
            elif current_trade.position_type == "short" and current_z >= strategy_params['stop_loss']:
                current_trade.status = "stopped"
                exit_condition = True
                
            # Take profit conditions
            if current_trade.position_type == "long" and current_z >= strategy_params['take_profit']:
                exit_condition = True
            elif current_trade.position_type == "short" and current_z <= -strategy_params['take_profit']:
                exit_condition = True
                
            # Max duration condition
            trade_duration = (timestamp - current_trade.entry_time).total_seconds() / 3600
            if trade_duration > strategy_params['max_trade_duration']:
                current_trade.status = "expired"
                exit_condition = True
                
            # Close trade if any exit condition met
            if exit_condition:
                current_trade.exit_time = timestamp
                current_trade.exit_price1 = price1
                current_trade.exit_price2 = price2
                
                # Calculate P&L
                fee = strategy_params['trading_fee']
                slippage = strategy_params['slippage']
                pnl = current_trade.calculate_pnl(fee, slippage)
                
                # Scale to position size
                position_size = strategy_params['position_size']
                dollar_pnl = pnl * position_size
                
                trades.append({
                    'entry_time': current_trade.entry_time,
                    'exit_time': current_trade.exit_time,
                    'symbol1': current_trade.symbol1,
                    'symbol2': current_trade.symbol2,
                    'position_type': current_trade.position_type,
                    'entry_price1': current_trade.entry_price1,
                    'entry_price2': current_trade.entry_price2,
                    'exit_price1': current_trade.exit_price1,
                    'exit_price2': current_trade.exit_price2,
                    'hedge_ratio': current_trade.hedge_ratio,
                    'status': current_trade.status,
                    'return': pnl,
                    'dollar_pnl': dollar_pnl,
                    'duration_hours': trade_duration
                })
                current_trade = None
    
    return trades

# ----------------------------------------------------------------------
# PERFORMANCE ANALYSIS
# ----------------------------------------------------------------------
def analyze_performance(trade_log):
    """Calculate performance metrics from trade log."""
    if not trade_log:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'total_return': 0
        }
    
    df = pd.DataFrame(trade_log)
    df['win'] = df['return'] > 0
    
    # Basic metrics
    total_trades = len(df)
    win_rate = df['win'].mean()
    avg_return = df['return'].mean()
    total_return = df['dollar_pnl'].sum()
    
    # Risk metrics
    returns = df['return']
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365*24)  # Annualized
    
    # Drawdown calculation
    equity = df['dollar_pnl'].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak)
    max_drawdown = drawdown.min()
    
    # Profit factor
    gross_profit = df[df['dollar_pnl'] > 0]['dollar_pnl'].sum()
    gross_loss = abs(df[df['dollar_pnl'] < 0]['dollar_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'total_return': total_return
    }

# ----------------------------------------------------------------------
# WALK-FORWARD BACKTEST
# ----------------------------------------------------------------------
def walkforward_backtest(config):
    """Run walkforward backtest across all intervals."""
    results = {}
    
    for interval in config['intervals']:
        print(f"\n=== Backtesting {interval} interval ===")
        interval_results = []
        
        # Load data for all symbols
        data_dict = {}
        for symbol in config['cryptos']:
            # In a real implementation, you'd load from actual data files
            data_dict[symbol] = load_crypto_data(None, symbol)
        
        # Merge data
        merged_data = merge_data(data_dict, interval)
        
        # Generate walkforward windows
        start_date = pd.Timestamp(config['start_date'])
        end_date = pd.Timestamp(config['end_date'])
        train_days = config['walkforward']['train_days']
        trade_days = config['walkforward']['trade_days']
        buffer_days = config['walkforward']['buffer_days']
        
        current_date = start_date
        while current_date < end_date:
            # Define windows
            train_end = current_date + pd.Timedelta(days=train_days)
            trade_start = train_end
            trade_end = trade_start + pd.Timedelta(days=trade_days)
            buffer_end = trade_end + pd.Timedelta(days=buffer_days)
            
            if buffer_end > end_date:
                break
                
            print(f"\nPeriod: {current_date.date()} to {trade_end.date()}")
            print(f"  Training: {current_date.date()} - {train_end.date()}")
            print(f"  Trading:  {trade_start.date()} - {trade_end.date()}")
            print(f"  Buffer:   {trade_end.date()} - {buffer_end.date()}")
            
            # Extract training data
            train_data = merged_data.loc[current_date:train_end]
            
            # Select pairs using training data
            selected_pairs = select_pairs(
                train_data, 
                config['metrics'], 
                config['selection']
            )
            
            print(f"  Selected {len(selected_pairs)} pairs:")
            for pair in selected_pairs:
                print(f"    {pair['symbol1']}/{pair['symbol2']} - ", end="")
                if 'cointegrated' in pair:
                    print(f"Cointegrated (p={pair['eg_pvalue']:.4f})")
                elif 'distance' in pair:
                    print(f"Distance={pair['distance']:.4f}")
                elif 'correlation' in pair:
                    print(f"Corr={pair['correlation']:.4f}")
            
            # Prepare trading data
            trade_data = merged_data.loc[trade_start:buffer_end]
            
            # Trade each selected pair
            period_trades = []
            for pair in selected_pairs:
                # Prepare strategy parameters
                pair_params = prepare_pair_strategy(
                    train_data, 
                    pair, 
                    config['strategy']
                )
                
                # Execute trading
                trades = execute_trading(
                    trade_data,
                    pair_params,
                    config['strategy']
                )
                
                # Add pair info to trades
                for trade in trades:
                    trade.update({
                        'pair': f"{pair['symbol1']}/{pair['symbol2']}",
                        'train_start': current_date.date(),
                        'train_end': train_end.date()
                    })
                
                period_trades.extend(trades)
            
            # Analyze performance for this period
            if period_trades:
                perf = analyze_performance(period_trades)
                print(f"  Performance: {perf['total_trades']} trades, "
                      f"Return: ${perf['total_return']:.2f}, "
                      f"Sharpe: {perf['sharpe_ratio']:.2f}")
                
                # Save results
                period_result = {
                    'start_date': current_date.date(),
                    'end_date': trade_end.date(),
                    'interval': interval,
                    'selected_pairs': [f"{p['symbol1']}/{p['symbol2']}" for p in selected_pairs],
                    'num_trades': perf['total_trades'],
                    'total_return': perf['total_return'],
                    'sharpe_ratio': perf['sharpe_ratio'],
                    'win_rate': perf['win_rate'],
                    'max_drawdown': perf['max_drawdown'],
                    'trades': period_trades
                }
                interval_results.append(period_result)
            
            # Move to next period
            current_date = trade_start + pd.Timedelta(days=1)
        
        results[interval] = interval_results
    
    return results

# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency Pairs Trading Backtester")
    parser.add_argument("-c", "--config", help="Path to config file (JSON)")
    parser.add_argument("-o", "--output", help="Output directory for results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Update output directory if provided
    if args.output:
        config['output_dir'] = args.output
    
    # Run backtest
    results = walkforward_backtest(config)
    
    # Save results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Save summary
    summary = []
    for interval, interval_results in results.items():
        for result in interval_results:
            summary.append({
                'interval': interval,
                'start_date': result['start_date'],
                'end_date': result['end_date'],
                'num_pairs': len(result['selected_pairs']),
                'num_trades': result['num_trades'],
                'total_return': result['total_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'win_rate': result['win_rate'],
                'max_drawdown': result['max_drawdown']
            })
    
    summary_df = pd.DataFrame(summary)
    summary_file = output_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary to {summary_file}")
    
    # Save trade log
    all_trades = []
    for interval, interval_results in results.items():
        for result in interval_results:
            all_trades.extend(result['trades'])
    
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_file = output_dir / "trades.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"Saved trade log to {trades_file}")
    else:
        print("No trades executed")

if __name__ == "__main__":
    main()