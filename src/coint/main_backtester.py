#!/usr/bin/env python3
"""
Main Backtester Entry Point
Cointegration-based Statistical Arbitrage Backtester
"""

import sys
import argparse
import warnings

from common_analysis import (
    setup_logging, load_config, add_common_arguments, list_data_files
)
from config_defaults import DEFAULT_CONFIG
from backtester_engine import CointegrationBacktester

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(
        description="Cointegration Statistical Arbitrage Backtester",
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
        log.info(f"Starting cointegration backtest for {args.interval} interval")
        backtester = CointegrationBacktester(config, repo_root, args.interval, log)
        backtester.run()
        
        log.info("Backtest completed successfully")
        return 0
        
    except Exception as e:
        log.exception(f"Backtest failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())