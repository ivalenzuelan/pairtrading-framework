#!/usr/bin/env python3
"""
Hurst Exponent-based Statistical Arbitrage Backtester
Integrated with common_analysis module for data loading and configuration
"""

import sys
import logging
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from typing import List, Dict, Tuple, Optional

# Import common utilities
from common_analysis import (
    setup_logging, load_config, load_interval_data, generate_walkforward_windows_strict,
    get_valid_pairs, extract_pair_data_from_window, calculate_common_metrics,
    save_results, create_analysis_stats, update_stats, log_final_stats,
    add_common_arguments, list_data_files, WindowPhase
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# CONSTANTS AND CONFIGURATION
# ----------------------------------------------------------------------
DEFAULT_CONFIG = {
    "cryptos": ["ETHUSDT", "BNBUSDT", "BTCUSDT", "MATICUSDT", "SHIBUSDT", "SANDUSDT", 
                "SOLUSDT", "GALAUSDT", "XRPUSDT", "AVAXUSDT", "DOTUSDT", "ADAUSDT", 
                "DOGEUSDT", "MANAUSDT", "FTMUSDT", "NEARUSDT", "TRXUSDT", "FILUSDT", 
                "LINKUSDT", "MBOXUSDT", "LTCUSDT", "ATOMUSDT", "CTXCUSDT", "CRVUSDT", 
                "EGLDUSDT", "EOSUSDT", "SUSHIUSDT", "ALICEUSDT", "AXSUSDT", "ICPUSDT"],
    "intervals": ["1m", "5m", "1h"],
    "start_date": "2022-01-01",
    "end_date": "2022-03-31",
    "forming_period": 5,             # Days for forming period
    "trading_period": 1,             # Days for trading period
    "buffer_period": 5,              # Max holding period
    "max_pairs_per_window": 5,       # Max pairs selected per forming period
    "allocation_per_pair": 1000,     # Dollar allocation per pair ($500 per leg)
    "taker_fee": 0.0004,             # 0.04% taker fee as per Binance
    "slippage_rate": 0.0005,         # Slippage rate (0.05%)
    "min_daily_volume": 1000000,     # Minimum daily volume filter (USD)
    "hurst_threshold": 0.4,          # Hurst exponent threshold (H < 0.4 indicates mean reversion)
    "entry_z": 2.0,                  # Z-score entry threshold
    "exit_z": 0.0,                   # Z-score exit threshold (SMA level)
    "min_rows_per_window": 100,      # Minimum data points required
    "min_spread_std": 0.01           # Minimum spread standard deviation to consider
}

# ----------------------------------------------------------------------
# HURST EXPONENT CALCULATION
# ----------------------------------------------------------------------
def calculate_hurst_exponent(series: pd.Series, max_lag: int = 50) -> float:
    """
    Calculate the Hurst exponent using rescaled range (R/S) analysis.
    Returns the Hurst exponent and its standard error.
    """
    if len(series) < 100:
        return 0.5  # Default to random walk if insufficient data
    
    lags = range(10, min(len(series)//2, max_lag))
    rs_values = []
    n_values = []
    
    for lag in lags:
        # Split the series into chunks of size = lag
        chunks = [series.values[i:i+lag] for i in range(0, len(series), lag)]
        if len(chunks[-1]) < lag//2:  # Discard last chunk if too small
            chunks = chunks[:-1]
        
        chunk_rs = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            # Calculate cumulative deviation from mean
            mean = np.mean(chunk)
            deviations = chunk - mean
            cumulative_dev = np.cumsum(deviations)
            
            # Calculate range
            r = np.max(cumulative_dev) - np.min(cumulative_dev)
            # Calculate standard deviation
            s = np.std(chunk)
            if s > 0:
                chunk_rs.append(r / s)
        
        if chunk_rs:
            rs_values.append(np.mean(chunk_rs))
            n_values.append(lag)
    
    if len(rs_values) < 3:
        return 0.5  # Not enough points to compute
    
    # Linear regression on log values
    log_rs = np.log(rs_values)
    log_n = np.log(n_values)
    
    # Calculate Hurst exponent as slope of log(R/S) vs log(n)
    slope, _, _, _, _ = linregress(log_n, log_rs)
    return slope

# ----------------------------------------------------------------------
# PAIR SCREENING (STAGE 0 & 1)
# ----------------------------------------------------------------------
def screen_mean_reverting_pairs(data: pd.DataFrame, config: dict, window: WindowPhase, 
                                log: logging.Logger) -> Tuple[List[dict], List[dict]]:
    """Screen for mean-reverting pairs using Hurst exponent"""
    results = []
    valid_pairs = []
    
    # Extract forming period data
    forming_data = data.loc[window.train_start:window.train_end]
    
    if forming_data.empty:
        log.warning(f"No data for forming period {window.train_start} to {window.train_end}")
        return results, valid_pairs
    
    # Get all valid pairs
    crypto_pairs = get_valid_pairs(forming_data, config["cryptos"])
    
    log.debug(f"Found {len(crypto_pairs)} potential pairs for window {window.iteration}")
    
    for (symbol1, col1), (symbol2, col2) in crypto_pairs:
        # Skip if column names are invalid
        if not col1 or not col2:
            log.debug(f"Skipping pair with empty column: {symbol1}/{symbol2}")
            continue
            
        try:
            # Extract slice for this pair
            pair_slice = forming_data[[col1, col2]].copy()
            
            # Calculate data completeness
            total_points = len(pair_slice)
            non_na_points = pair_slice.notna().all(axis=1).sum()
            coverage = non_na_points / total_points if total_points > 0 else 0
            
            # Fill missing values within the forming period
            pair_slice = pair_slice.ffill().bfill()
            
            # Remove any remaining NA (shouldn't be any after fill)
            pair_df = pair_slice.dropna()
            actual_points = len(pair_df)
            
            # Check if we have enough data
            min_rows = config.get("min_rows_per_window", 100)
            if actual_points < min_rows:
                log.debug(
                    f"Skipping {symbol1}/{symbol2}: insufficient data "
                    f"(required: {min_rows}, found: {actual_points}) "
                    f"(before fill: {non_na_points}/{total_points})"
                )
                continue
                
            # Log data quality
            log.debug(
                f"Processing {symbol1}/{symbol2}: "
                f"{actual_points}/{total_points} points "
                f"({coverage:.1%} coverage)"
            )
            
            # Get price series
            s1 = pair_df[col1]
            s2 = pair_df[col2]
            
            # Estimate hedge ratio
            try:
                slope, intercept, _, _, _ = linregress(s2, s1)
            except Exception as e:
                log.debug(f"Regression failed for {symbol1}/{symbol2}: {e}")
                continue
            
            # Calculate spread
            spread = s1 - slope * s2 - intercept
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Skip if spread is too stable
            if spread_std < config["min_spread_std"]:
                log.debug(
                    f"Skipping {symbol1}/{symbol2}: spread too stable "
                    f"(std={spread_std:.6f})"
                )
                continue
            
            # Calculate Hurst exponent
            hurst_exp = calculate_hurst_exponent(spread)
            
            # Check if mean-reverting (H < threshold)
            mean_reverting = hurst_exp < config["hurst_threshold"]
            
            if not mean_reverting:
                log.debug(
                    f"Skipping {symbol1}/{symbol2}: not mean-reverting "
                    f"(H={hurst_exp:.4f})"
                )
                continue
            
            pair_info = {
                "pair": f"{symbol1}/{symbol2}",
                "asset1": symbol1,
                "asset2": symbol2,
                "hedge_ratio": slope,
                "intercept": intercept,
                "mean": spread_mean,
                "std": spread_std,
                "hurst": hurst_exp,
                "window_iteration": window.iteration,
                "data_points": actual_points,
                "coverage": coverage
            }
            
            results.append(pair_info)
            valid_pairs.append(pair_info)
            log.info(
                f"Found mean-reverting pair: {symbol1}/{symbol2} "
                f"(H={hurst_exp:.4f}, std={spread_std:.4f}, "
                f"data={actual_points}/{total_points} points)"
            )
            
        except Exception as e:
            log.error(f"Error processing pair {symbol1}/{symbol2}: {e}")
            continue
    
    # Return only top pairs (max 5 per window)
    max_pairs = min(len(valid_pairs), config["max_pairs_per_window"])
    return results, valid_pairs[:max_pairs]

# ----------------------------------------------------------------------
# TRADING EXECUTION (STAGE 2)
# ----------------------------------------------------------------------
class PairTrader:
    """Manages trading for a single mean-reverting pair"""
    def __init__(self, pair_info: dict, config: dict, interval: str, log: logging.Logger):
        # Initialize from pair screening results
        self.asset1 = pair_info["asset1"]
        self.asset2 = pair_info["asset2"]
        self.hedge_ratio = pair_info["hedge_ratio"]
        self.intercept = pair_info["intercept"]
        self.spread_mean = pair_info["mean"]
        self.spread_std = pair_info["std"]
        
        # Store configuration and dependencies
        self.config = config
        self.interval = interval
        self.log = log
        
        # Initialize trading state
        self.position = None  # 'long', 'short', or None
        self.entry_time = None
        self.entry_price = None  # (asset1_price, asset2_price)
        self.entry_z = None
        self.trade_history = []
        
    def calculate_spread(self, asset1_price: float, asset2_price: float) -> float:
        """Calculate spread using hedge ratio and intercept"""
        return asset1_price - self.hedge_ratio * asset2_price - self.intercept
    
    def calculate_z_score(self, spread: float) -> float:
        """Calculate z-score of spread"""
        return (spread - self.spread_mean) / self.spread_std if self.spread_std > 0 else 0
    
    def check_signal(self, z_score: float, timestamp: pd.Timestamp) -> Optional[str]:
        """Generate trading signals based on Bollinger Bands strategy"""
        # Exit conditions
        if self.position and self.entry_time is not None:
            holding_days = (timestamp - self.entry_time).days
            exit_condition = (
                (self.position == "long" and z_score >= self.config["exit_z"]) or
                (self.position == "short" and z_score <= self.config["exit_z"]) or
                holding_days >= self.config["buffer_period"]
            )
            if exit_condition:
                return "exit"
        
        # Entry conditions (only if not in position)
        if not self.position:
            if z_score <= -self.config["entry_z"]:
                return "long"
            elif z_score >= self.config["entry_z"]:
                return "short"
        
        return None
    
    def execute_trade(self, signal: str, timestamp: pd.Timestamp, 
                      asset1_price: float, asset2_price: float, 
                      current_z_score: float) -> dict: 
        """Execute trade and record transaction"""
        # Determine quantities based on dollar allocation ($500 per leg)
        allocation = self.config["allocation_per_pair"] / 2
        asset1_qty = allocation / asset1_price
        asset2_qty = allocation / asset2_price
        
        if signal == "long":
            # Buy spread: Buy asset1, Sell asset2
            self.position = "long"
            asset1_side = "buy"
            asset2_side = "sell"
        elif signal == "short":
            # Sell spread: Sell asset1, Buy asset2
            self.position = "short"
            asset1_side = "sell"
            asset2_side = "buy"
        else:  # Exit
            # Reverse current position
            asset1_side = "sell" if self.position == "long" else "buy"
            asset2_side = "buy" if self.position == "long" else "sell"
            self.position = None
        
        # Apply slippage
        slippage = self.config["slippage_rate"]
        asset1_exec = asset1_price * (1 + slippage) if asset1_side == "buy" else asset1_price * (1 - slippage)
        asset2_exec = asset2_price * (1 + slippage) if asset2_side == "buy" else asset2_price * (1 - slippage)
        
        # Calculate costs
        asset1_cost = asset1_qty * asset1_exec
        asset2_cost = asset2_qty * asset2_exec
        fee = (asset1_cost + asset2_cost) * self.config["taker_fee"]
        
        # Record trade
        trade = {
            "timestamp": timestamp,
            "signal": signal,
            "position": self.position,
            "asset1": self.asset1,
            "asset1_qty": asset1_qty,
            "asset1_side": asset1_side,
            "asset1_price": asset1_exec,
            "asset2": self.asset2,
            "asset2_qty": asset2_qty,
            "asset2_side": asset2_side,
            "asset2_price": asset2_exec,
            "fee": fee,
            "z_score": self.entry_z if signal == "exit" else current_z_score
        }
        self.trade_history.append(trade)
        
        # Update position state
        if signal in ("long", "short"):
            self.entry_time = timestamp
            self.entry_price = (asset1_price, asset2_price)
            self.entry_z = current_z_score
        else:
            self.entry_time = None
            self.entry_price = None
            self.entry_z = None
            
        return trade

# ----------------------------------------------------------------------
# PORTFOLIO MANAGEMENT
# ----------------------------------------------------------------------
class Portfolio:
    """Manages overall portfolio state"""
    def __init__(self, initial_cash: float = 1000000):
        self.cash = initial_cash
        self.positions = {}  # Format: {'asset': {'qty': quantity}}
        self.value_history = []  # (timestamp, total_value)
        self.trades = []
        
    def update_position(self, asset: str, qty: float):
        """Update position for an asset"""
        asset_key = f"{asset}_position"
        
        if asset_key not in self.positions:
            self.positions[asset_key] = {"qty": 0}
        
        self.positions[asset_key]["qty"] += qty
        
        # Clean up zero positions
        if abs(self.positions[asset_key]["qty"]) < 1e-6:
            del self.positions[asset_key]
    
    def calculate_value(self, prices: dict) -> float:
        """Calculate current portfolio value"""
        total_value = self.cash
        for asset, position in self.positions.items():
            crypto = asset.split('_')[0]
            if crypto in prices:
                total_value += position["qty"] * prices[crypto]
        return total_value
    
    def record_value(self, timestamp: pd.Timestamp, prices: dict):
        """Record portfolio value at a specific timestamp"""
        current_value = self.calculate_value(prices)
        self.value_history.append((timestamp, current_value))

# ----------------------------------------------------------------------
# BACKTESTING ENGINE
# ----------------------------------------------------------------------
class HurstBacktester:
    """Main backtesting engine for Hurst-based mean reversion strategy"""
    def __init__(self, config: dict, repo_root: Path, interval: str, log: logging.Logger):
        self.config = config
        self.repo_root = repo_root
        self.interval = interval
        self.log = log
        self.data = None
        self.portfolio = Portfolio()
        self.active_traders = {}  # pair: PairTrader
        self.results = []
        
        # Set adaptive minimum rows based on interval
        self.min_rows_map = {
            '1m': 5 * 24 * 60 * 0.8,   # 5 days (80% of 7200 minutes)
            '5m': 5 * 24 * 12 * 0.8,   # 5 days (80% of 1440 periods)
            '1h': 5 * 24 * 0.8          # 5 days (80% of 120 hours)
        }

        if interval == '1h':
            self.min_rows_map['1h'] = 97
        self.min_rows = self.min_rows_map.get(interval, config["min_rows_per_window"])
        self.log.info(f"Using adaptive min_rows: {self.min_rows} for {interval} interval")
        

    def load_data(self):
        """Load data for the specified interval with gap handling"""
        try:
            self.data = load_interval_data(
                self.repo_root, self.config["cryptos"], self.interval, self.log
            )
            
            # Resample to ensure consistent frequency
            freq_map = {'1m': '1T', '5m': '5T', '1h': '1H'}
            freq = freq_map.get(self.interval)
            
            if freq:
                # Create complete time index covering entire date range
                full_index = pd.date_range(
                    start=self.config["start_date"],
                    end=self.config["end_date"],
                    freq=freq
                )
                
                # Reindex to ensure all time periods are present
                self.data = self.data.reindex(full_index)
                
                # Fill missing values
                self.data = self.data.ffill().bfill()
                
                na_count = self.data.isna().sum().sum()
                if na_count > 0:
                    self.log.warning(f"Still have {na_count} NA values after filling")
            
            # Validate data coverage
            for crypto in self.config["cryptos"]:
                if crypto in self.data.columns:
                    non_na = self.data[crypto].count()
                    total = len(self.data)
                    coverage = non_na / total
                    status = "OK" if coverage > 0.95 else "LOW"
                    self.log.info(f"{status} - {crypto}: {non_na}/{total} rows ({coverage:.1%})")
            
            self.log.info(f"Loaded data for {self.interval}: {self.data.shape[0]} rows")
            self.log.info(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            return True
        except Exception as e:
            self.log.error(f"Failed to load data for {self.interval}: {e}")
            return False

    
    def run(self):
        """Execute backtest over the entire date range"""
        if not self.load_data():
            return
        
        # Generate walk-forward windows
        windows = generate_walkforward_windows_strict(
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
            training_days=self.config["forming_period"],
            trading_days=self.config["trading_period"],
            buffer_days=self.config["buffer_period"],
            step_days=1,
            log=self.log
        )
        
        if not windows:
            self.log.error("No valid windows generated")
            return
        
        stats = create_analysis_stats()
        
        for window in windows:
            self.log.info(f"Processing window {window.iteration}: "
                         f"{window.train_start.date()} to {window.buffer_end.date()}")

            window_config = self.config.copy()
            window_config["min_rows_per_window"] = self.min_rows
            
            # Screen pairs for this forming period
            try:
                _, valid_pairs = screen_mean_reverting_pairs(
                    self.data, window_config, window, self.log
                )
            except Exception as e:
                self.log.error(f"Error screening pairs for window {window.iteration}: {e}")
                continue
            
            if not valid_pairs:
                self.log.info(f"No valid pairs found for window {window.iteration}")
                continue
            
            # Initialize traders for valid pairs
            for pair_info in valid_pairs:
                required_keys = ["asset1", "asset2", "hedge_ratio", "intercept", "mean", "std"]
                missing_keys = [k for k in required_keys if k not in pair_info]
                if missing_keys:
                    self.log.warning(f"Skipping pair {pair_info.get('asset1', '?')}/{pair_info.get('asset2', '?')}: missing keys {missing_keys}")
                    continue
                self.log.debug(f"About to instantiate PairTrader with pair_info: {pair_info}")
                try:
                    trader = PairTrader(
                        pair_info, 
                        self.config, 
                        self.interval, 
                        self.log
                    )
                    self.active_traders[pair_info["pair"]] = trader
                    update_stats(stats, "tested")
                    self.log.info(f"Added trader for {pair_info['pair']}")
                except Exception as e:
                    self.log.error(f"Error creating trader for {pair_info['pair']}: {e}")
            
            # Execute trading for this window
            self.execute_trading_window(window)
            
            # Clear traders after window completes
            self.active_traders = {}
        
        # Final portfolio recording
        if self.data.index[-1] not in [v[0] for v in self.portfolio.value_history]:
            last_prices = {col: self.data[col].iloc[-1] for col in self.data.columns}
            self.portfolio.record_value(self.data.index[-1], last_prices)
        
        # Save results
        self.save_results()
        log_final_stats(stats, self.interval, self.log)
    
    def execute_trading_window(self, window: WindowPhase):
        """Execute trading for a specific window"""
        # Get trading period data (including buffer)
        try:
            def get_nearest_index(target):
                idx = self.data.index
                try:
                    return idx.get_loc(target)
                except KeyError:
                    return idx.searchsorted(target)

            start_idx = get_nearest_index(window.trade_start)
            end_idx = get_nearest_index(window.buffer_end)
            window_data = self.data.iloc[start_idx:end_idx+1]
        except KeyError:
            self.log.error(f"Window dates out of range: {window.trade_start} to {window.buffer_end}")
            return
        
        for timestamp, row in window_data.iterrows():
            # Get current prices - use all available columns
            current_prices = {col: row[col] for col in self.data.columns}
            
            # Update portfolio value
            self.portfolio.record_value(timestamp, current_prices)
            
            # Process each active trader
            for pair, trader in list(self.active_traders.items()):
                # Get prices for this pair
                asset1_price = row.get(trader.asset1, np.nan)
                asset2_price = row.get(trader.asset2, np.nan)
                
                if np.isnan(asset1_price) or np.isnan(asset2_price):
                    self.log.debug(f"Missing price data for {pair} at {timestamp}")
                    continue
                
                # Calculate spread and z-score
                spread = trader.calculate_spread(asset1_price, asset2_price)
                z_score = trader.calculate_z_score(spread)
                
                # Check for trading signal
                signal = trader.check_signal(z_score, timestamp)
                if not signal:
                    continue
                
                # Execute trade
                try:
                    trade = trader.execute_trade(signal, timestamp, asset1_price, asset2_price, z_score)
                    self.portfolio.trades.append(trade)
                    self.log.info(f"Executed {signal} trade for {pair} at z={z_score:.2f}")
                    
                    # Update portfolio
                    self.process_trade(trade)
                    
                    # Remove trader if position closed
                    if signal == "exit":
                        del self.active_traders[pair]
                        self.log.info(f"Closed position for {pair}")
                except Exception as e:
                    self.log.error(f"Error executing trade for {pair}: {e}")
    
    def process_trade(self, trade: dict):
        """Update portfolio based on trade execution"""
        # Update cash
        cost = 0
        
        # Process asset1 transaction
        asset1_qty = trade["asset1_qty"]
        asset1_exec = trade["asset1_price"]
        asset1_side = trade["asset1_side"]
        
        if asset1_side == "buy":
            cost -= asset1_qty * asset1_exec
            self.portfolio.update_position(trade["asset1"], asset1_qty)
        else:  # sell
            cost += asset1_qty * asset1_exec
            self.portfolio.update_position(trade["asset1"], -asset1_qty)
        
        # Process asset2 transaction
        asset2_qty = trade["asset2_qty"]
        asset2_exec = trade["asset2_price"]
        asset2_side = trade["asset2_side"]
        
        if asset2_side == "buy":
            cost -= asset2_qty * asset2_exec
            self.portfolio.update_position(trade["asset2"], asset2_qty)
        else:  # sell
            cost += asset2_qty * asset2_exec
            self.portfolio.update_position(trade["asset2"], -asset2_qty)
        
        # Apply fees
        self.portfolio.cash += cost - trade["fee"]
    
    def save_results(self):
        """Save backtest results to files"""
        output_dir = self.repo_root / "reports" / "hurst" / self.interval
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save equity curve
        equity_df = pd.DataFrame(self.portfolio.value_history, columns=["timestamp", "value"])
        equity_path = output_dir / "equity_curve.csv"
        equity_df.to_csv(equity_path, index=False)
        self.log.info(f"Saved equity curve to {equity_path}")
        
        # Save trades
        if self.portfolio.trades:
            trades_df = pd.DataFrame(self.portfolio.trades)
            trades_path = output_dir / "trades.csv"
            trades_df.to_csv(trades_path, index=False)
            self.log.info(f"Saved {len(trades_df)} trades to {trades_path}")
        else:
            self.log.warning("No trades executed during backtest")
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.log.info(f"Saved performance metrics to {metrics_path}")
    
    def calculate_performance_metrics(self) -> dict:
        """Calculate key performance metrics"""
        if not self.portfolio.value_history:
            return {}
        
        # Extract portfolio values
        values = [v for _, v in self.portfolio.value_history]
        returns = pd.Series(values).pct_change().dropna()
        initial_value = values[0]
        final_value = values[-1]
        
        # Calculate returns
        cumulative_return = (final_value / initial_value) - 1
        
        # Calculate volatility (annualized)
        volatility = returns.std() * np.sqrt(365.25 * 24)  # Assuming minute data
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365.25 * 24) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Trade analysis
        if self.portfolio.trades:
            trades = pd.DataFrame(self.portfolio.trades)
            
            # Calculate profit for each trade
            trades["profit"] = 0
            for i, trade in trades.iterrows():
                # Calculate P&L for the trade
                entry_multiplier = -1 if trade["signal"] == "exit" else 1
                asset1_pnl = trade["asset1_qty"] * (trade["asset1_price"] if trade["asset1_side"] == "sell" else -trade["asset1_price"]) * entry_multiplier
                asset2_pnl = trade["asset2_qty"] * (trade["asset2_price"] if trade["asset2_side"] == "sell" else -trade["asset2_price"]) * entry_multiplier
                trades.at[i, "profit"] = asset1_pnl + asset2_pnl - trade["fee"]
            
            wins = trades[trades["profit"] > 0]
            win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
            avg_win = wins["profit"].mean() if len(wins) > 0 else 0
            losses = trades[trades["profit"] <= 0]
            avg_loss = losses["profit"].mean() if len(losses) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        return {
            "cumulative_return": cumulative_return,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.portfolio.trades) if self.portfolio.trades else 0,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "final_value": final_value,
            "initial_value": initial_value
        }

# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Hurst Exponent Mean Reversion Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = add_common_arguments(parser)
    parser.add_argument(
        "--interval", 
        choices=["1m", "5m", "1h"], 
        default="1h",
        help="Data interval to use for backtesting"
    )
    args = parser.parse_args()
    
    # Setup logging
    log = setup_logging("backtester", args.verbose)
    
    try:
        # Load configuration
        config, repo_root = load_config(args.config, __file__)
        
        # Update with default values
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
        
        # Ensure cryptos match data columns
        config["cryptos"] = [crypto.upper() for crypto in config["cryptos"]]
        
        # List files if requested
        if args.list_files:
            list_data_files(repo_root, log)
            return 0
        
        # Run backtest for specified interval
        log.info(f"Starting Hurst backtest for {args.interval} interval")
        backtester = HurstBacktester(config, repo_root, args.interval, log)
        backtester.run()
        
        log.info("Backtest completed successfully")
        return 0
        
    except Exception as e:
        log.exception(f"Backtest failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())