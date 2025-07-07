#!/usr/bin/env python3
"""
Backtesting Engine Module
Main backtesting logic for cointegration strategy
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional

from common_analysis import (
    load_interval_data, generate_walkforward_windows_strict,
    create_analysis_stats, update_stats, log_final_stats, WindowPhase
)
from pair_screening import screen_cointegrated_pairs
from pair_trader import PairTrader
from portfolio_manager import Portfolio

class CointegrationBacktester:
    """Main backtesting engine for cointegration strategy"""
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
        self.min_rows = self.min_rows_map.get(interval, config["min_rows_per_window"])
        self.log.info(f"Using adaptive min_rows: {self.min_rows} for {interval} interval")
        
    def load_data(self) -> bool:
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
                _, valid_pairs = screen_cointegrated_pairs(
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
                try:
                    trader = PairTrader(
                        pair_info, 
                        self.config, 
                        self.interval, 
                        self.log
                    )
                    self.active_traders[pair_info["pair"]] = trader
                    update_stats(stats, "tested")
                    self.log.info(f"Added trader for {pair_info['pair']} ({pair_info['leader']}->{pair_info['follower']})")
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
                leader_price = row.get(trader.leader, np.nan)
                follower_price = row.get(trader.follower, np.nan)
                
                if np.isnan(leader_price) or np.isnan(follower_price):
                    self.log.debug(f"Missing price data for {pair} at {timestamp}")
                    continue
                
                # Calculate spread and z-score (using RATIO spread)
                spread = trader.calculate_spread(leader_price, follower_price)
                z_score = trader.calculate_z_score(spread)
                
                # Check for trading signal
                signal = trader.check_signal(z_score, timestamp)
                if not signal:
                    continue
                
                # Execute trade
                try:
                    trade = trader.execute_trade(signal, timestamp, leader_price, follower_price, z_score)
                    self.portfolio.trades.append(trade)
                    
                    if signal in ("stop_loss", "time_stop"):
                        log_msg = f"STOP-LOSS triggered ({signal}) for {pair} at z={z_score:.2f}"
                    else:
                        log_msg = f"Executed {signal} trade for {pair} at z={z_score:.2f}"
                    self.log.info(log_msg)
                    
                    # Update portfolio
                    self.portfolio.process_trade(trade)
                    
                    # Remove trader if position closed
                    if signal in ("exit", "stop_loss", "time_stop"):
                        del self.active_traders[pair]
                        self.log.info(f"Closed position for {pair}")
                except Exception as e:
                    self.log.error(f"Error executing trade for {pair}: {e}")
    
    def save_results(self):
        """Save backtest results to files"""
        output_dir = self.repo_root / "reports" / "cointegration" / self.interval
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
        metrics = self.portfolio.calculate_performance_metrics()
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.log.info(f"Saved performance metrics to {metrics_path}")
