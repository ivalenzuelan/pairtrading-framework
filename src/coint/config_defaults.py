#!/usr/bin/env python3
"""
Configuration Defaults Module
Default configuration values for the cointegration backtester
"""

DEFAULT_CONFIG = {
    "cryptos": ["ETHUSDT", "BNBUSDT", "BTCUSDT", "MATICUSDT", "SHIBUSDT", "SANDUSDT", 
                "SOLUSDT", "GALAUSDT", "XRPUSDT", "AVAXUSDT", "DOTUSDT", "ADAUSDT", 
                "DOGEUSDT", "MANAUSDT", "FTMUSDT", "NEARUSDT", "TRXUSDT", "FILUSDT", 
                "LINKUSDT", "MBOXUSDT", "LTCUSDT", "ATOMUSDT", "CTXCUSDT", "CRVUSDT", 
                "EGLDUSDT", "EOSUSDT", "SUSHIUSDT", "ALICEUSDT", "AXSUSDT", "ICPUSDT"],
    "intervals": ["1m", "5m", "1h"],
    "start_date": "2022-01-01",
    "end_date": "2022-01-31",
    "forming_period": 5,             # Days for forming period
    "trading_period": 1,             # Days for trading period
    "buffer_period": 5,              # Max holding period
    "max_pairs_per_window": 5,       # Max pairs selected per forming period
    "base_allocation": 1000,         # Base dollar allocation
    "taker_fee": 0.0004,             # 0.04% taker fee as per Binance
    "slippage_rate": 0.0005,         # Slippage rate (0.05%)
    "min_daily_volume": 1000000,     # Minimum daily volume filter (USD)
    "eg_p_threshold": 0.05,          # Engle-Granger p-value threshold
    "johansen_conf_level": 0.95,     # Johansen test confidence level
    "gc_lags": 4,                    # Lags for Granger causality test
    "gc_p_threshold": 0.05,          # Granger causality p-value threshold
    "entry_z": 2.0,                  # Z-score entry threshold
    "exit_z": 0.5,                   # Z-score exit threshold (0.5σ)
    "stop_loss_z": 3.0,              # Stop-loss threshold (3.0σ)
    "min_rows_per_window": 100,      # Minimum data points required
    "min_spread_std": 0.01,
    "gc_lags": 4,
    "gc_p_threshold": 0.05,
    "vecm_rank": "auto"         # Minimum spread standard deviation to consider
}