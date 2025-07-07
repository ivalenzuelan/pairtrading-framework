#!/usr/bin/env python3
"""
Correlation-based pair trading analysis.
Uses common utilities for data loading and processing.
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller

# Import common utilities
from common_analysis import (
    setup_logging, load_config, load_interval_data, generate_windows,
    get_valid_pairs, extract_pair_data, preprocess_series, check_data_quality,
    calculate_common_metrics, save_results, create_analysis_stats, 
    update_stats, log_final_stats, add_common_arguments, list_data_files
)

# ----------------------------------------------------------------------
# CORRELATION-SPECIFIC FUNCTIONS
# ----------------------------------------------------------------------
def calculate_correlation_robust(s1, s2, pair_tag="", preprocess_method='returns', log=None):
    """Robust correlation calculation with preprocessing and error handling."""
    try:
        # Preprocess data
        s1_proc, s2_proc = preprocess_series(s1, s2, preprocess_method)
        
        # Check data quality
        is_valid, issues = check_data_quality(s1_proc, s2_proc, pair_tag)
        if not is_valid:
            if log:
                log.debug(f"Data quality issues for {pair_tag}: {', '.join(issues)}")
            return np.nan, np.nan, np.nan
        
        # Ensure we have enough data after preprocessing
        if len(s1_proc) < 30:
            if log:
                log.debug(f"Insufficient data after preprocessing for {pair_tag}: {len(s1_proc)} points")
            return np.nan, np.nan, np.nan
        
        # Calculate Pearson and Spearman correlations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson_corr, pearson_p = pearsonr(s1_proc.values, s2_proc.values)  # Use .values to avoid index issues
            spearman_corr, spearman_p = spearmanr(s1_proc.values, s2_proc.values)
            
            # Calculate rolling correlation stability
            rolling_corr = s1_proc.rolling(window=min(21, len(s1_proc)), min_periods=1).corr(s2_proc)
            corr_stability = rolling_corr.std() if not rolling_corr.empty else np.nan
            
            return pearson_corr, spearman_corr, corr_stability
            
    except Exception as exc:
        if log:
            log.debug(f"Correlation failed for {pair_tag}: {exc}")
        return np.nan, np.nan, np.nan

def check_stationarity(series, pair_tag="", log=None):
    """Check if series is stationary using ADF test."""
    try:
        series = series.dropna()
        if len(series) < 10:  # Minimum samples for ADF test
            return False
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(series)
            return result[1] < 0.05  # p-value < 0.05 indicates stationarity
    except Exception as exc:
        if log:
            log.debug(f"Stationarity check failed for {pair_tag}: {exc}")
        return False

def analyze_correlation_pair(pair_df, metadata, config, log):
    """Analyze a single pair for correlation-based trading."""
    close1, close2 = metadata["close1"], metadata["close2"]
    symbol1, symbol2 = metadata["symbol1"], metadata["symbol2"]
    pair_tag = f"{symbol1}-{symbol2} {metadata['window_start'].date()}→{metadata['window_end'].date()}"
    
    # Calculate returns-based correlation
    pearson, spearman, corr_stability = calculate_correlation_robust(
        pair_df[close1], pair_df[close2], 
        pair_tag, 
        'returns', log
    )
    
    # Fallback to price correlation if returns fail
    if np.isnan(pearson):
        pearson, spearman, corr_stability = calculate_correlation_robust(
            pair_df[close1], pair_df[close2], 
            pair_tag, 
            'prices', log
        )
        method_used = 'prices'
    else:
        method_used = 'returns'
    
    if np.isnan(pearson):
        if log:
            log.debug(f"Skipping pair {pair_tag} - correlation calculation failed")
        return None

    # Check stationarity of spread (price difference)
    spread = pair_df[close1] - pair_df[close2]
    spread_stationary = check_stationarity(spread, pair_tag, log)
    
    # Calculate common metrics
    common_metrics = calculate_common_metrics(pair_df[close1], pair_df[close2])
    
    # Build result
    result = {
        "method": "correlation",
        "interval": metadata.get("interval", ""),
        "window_start": metadata["window_start"].date().isoformat(),
        "window_end": metadata["window_end"].date().isoformat(),
        "asset1": symbol1,
        "asset2": symbol2,
        "rows": metadata["rows"],
        
        # Correlation metrics
        "pearson_corr": round(pearson, 4),
        "spearman_corr": round(spearman, 4),
        "corr_stability": round(corr_stability, 4) if not np.isnan(corr_stability) else np.nan,
        "spread_stationary": spread_stationary,
        "preprocessing_method": method_used,
        
        # Use absolute correlation as similarity score
        "similarity_score": round(abs(pearson), 4),
    }
    
    # Add common metrics
    result.update(common_metrics)
    
    return result

# ----------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------------------------------------------------
def analyze_interval(config, repo_root, interval, log):
    """Analyze a single interval for correlated pairs."""
    log.info(f"▶ Analyzing interval {interval} for correlations...")
    stats = create_analysis_stats()
    
    try:
        data = load_interval_data(repo_root, config["cryptos"], interval, log)
    except (FileNotFoundError, ValueError) as e:
        log.error(f"Failed to load data for {interval}: {e}")
        return None

    results = []
    windows = generate_windows(
        config["start_date"], config["end_date"],
        config["window_days"], config["step_days"]
    )
    
    # Get all valid pairs
    valid_pairs = get_valid_pairs(data, config["cryptos"])

    for w_start, w_end in windows:
        for (symbol1, col1), (symbol2, col2) in valid_pairs:
            # Extract pair data for this window
            pair_df, metadata = extract_pair_data(
                data, (symbol1, col1), (symbol2, col2),
                w_start, w_end, config["min_rows_per_window"]
            )
            
            if pair_df is None:
                update_stats(stats, "too_short")
                continue

            # Add interval info to metadata
            metadata["interval"] = interval
            
            update_stats(stats, "tested")
            
            # Analyze this pair
            result = analyze_correlation_pair(pair_df, metadata, config, log)
            
            if result is None:
                update_stats(stats, "failed")
                continue
            
            update_stats(stats, "successful")
            if result["pearson_corr"] > config.get("min_correlation", 0.7):
                update_stats(stats, "high_correlation")
            
            results.append(result)

    # Save results
    if results:
        output_path = save_results(
            results, 
            repo_root / "reports" / "correlation",
            f"correlation_{interval}.csv",
            log
        )
        
        log_final_stats(stats, interval, log, f"→ {output_path}")
        return output_path
    else:
        log.warning(f"[{interval}] No valid results generated")
        return None

# ----------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Correlation analysis for cryptocurrency pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser = add_common_arguments(parser)
    args = parser.parse_args()
    
    # Setup logging
    log = setup_logging("correlation", args.verbose)
    
    try:
        config, repo_root = load_config(args.config, __file__)
        
        # List files if requested
        if args.list_files:
            list_data_files(repo_root, log)
            return 0
        
        # Run analysis
        result_paths = []
        for interval in config["intervals"]:
            result_path = analyze_interval(config, repo_root, interval, log)
            if result_path:
                result_paths.append(result_path)
        
        if result_paths:
            log.info("Correlation analysis completed")
            for path in result_paths:
                log.info(f"  → {path}")
            return 0
        else:
            log.error("No output files generated")
            return 1
            
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        if args.verbose:
            log.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())