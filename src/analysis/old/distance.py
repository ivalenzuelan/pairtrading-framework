#!/usr/bin/env python3
"""
Distance-based pair trading analysis.
Uses common utilities for data loading and processing.
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Import common utilities
from common_analysis import (
    setup_logging, load_config, load_interval_data, generate_windows,
    get_valid_pairs, extract_pair_data, preprocess_series, check_data_quality,
    calculate_common_metrics, save_results, create_analysis_stats, 
    update_stats, log_final_stats, add_common_arguments, list_data_files
)

# ----------------------------------------------------------------------
# DISTANCE-SPECIFIC FUNCTIONS
# ----------------------------------------------------------------------
def calculate_distance_robust(s1, s2, pair_tag="", preprocess_method='standardize', log=None):
    """Robust distance calculation with preprocessing and error handling."""
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
        
        # Calculate Euclidean distance
        euclidean_dist = np.sqrt(np.nansum((s1_proc - s2_proc)**2))
        
        # Calculate mean absolute deviation (MAD) as alternative distance metric
        mad_dist = np.nanmean(np.abs(s1_proc - s2_proc))
        
        # Calculate rolling distance stability
        rolling_dist = s1_proc.rolling(window=min(21, len(s1_proc)), min_periods=1).apply(
            lambda x: np.sqrt(np.nansum((x - s2_proc.loc[x.index])**2)), raw=False
        )
        dist_stability = rolling_dist.std() if not rolling_dist.empty else np.nan
        
        return euclidean_dist, mad_dist, dist_stability
            
    except Exception as exc:
        if log:
            log.debug(f"Distance calculation failed for {pair_tag}: {exc}")
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

def analyze_distance_pair(pair_df, metadata, config, log):
    """Analyze a single pair for distance-based trading."""
    col1, col2 = metadata["col1"], metadata["col2"]
    symbol1, symbol2 = metadata["symbol1"], metadata["symbol2"]
    pair_tag = f"{symbol1}-{symbol2} {metadata['window_start'].date()}→{metadata['window_end'].date()}"
    
    # Calculate distance with standardization
    euclidean_dist, mad_dist, dist_stability = calculate_distance_robust(
        pair_df[col1], pair_df[col2], 
        pair_tag, 
        'standardize', log
    )
    
    # Fallback to log prices if standardization fails
    if np.isnan(euclidean_dist):
        euclidean_dist, mad_dist, dist_stability = calculate_distance_robust(
            pair_df[col1], pair_df[col2], 
            pair_tag, 
            'log_prices', log
        )
        method_used = 'log_prices'
    else:
        method_used = 'standardize'
    
    if np.isnan(euclidean_dist):
        if log:
            log.debug(f"Skipping pair {pair_tag} - distance calculation failed")
        return None

    # Check stationarity of spread (price difference)
    spread = pair_df[col1] - pair_df[col2]
    spread_stationary = check_stationarity(spread, pair_tag, log)
    
    # Calculate common metrics
    common_metrics = calculate_common_metrics(pair_df[col1], pair_df[col2])
    
    # Convert distance to similarity score (inverse relationship)
    # Using exponential decay: similarity = e^(-distance)
    similarity_score = np.exp(-euclidean_dist / config.get("distance_scale", 10))
    
    # Build result
    result = {
        "method": "distance",
        "interval": metadata.get("interval", ""),
        "window_start": metadata["window_start"].date().isoformat(),
        "window_end": metadata["window_end"].date().isoformat(),
        "asset1": symbol1,
        "asset2": symbol2,
        "rows": metadata["rows"],
        
        # Distance metrics
        "euclidean_dist": round(euclidean_dist, 4),
        "mad_dist": round(mad_dist, 4),
        "dist_stability": round(dist_stability, 4) if not np.isnan(dist_stability) else np.nan,
        "spread_stationary": spread_stationary,
        "preprocessing_method": method_used,
        
        # Similarity score (higher is more similar)
        "similarity_score": round(similarity_score, 4),
    }
    
    # Add common metrics
    result.update(common_metrics)
    
    return result

# ----------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------------------------------------------------
def analyze_interval(config, repo_root, interval, log):
    """Analyze a single interval for distance-based pairs."""
    log.info(f"▶ Analyzing interval {interval} for distance-based pairs...")
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
            result = analyze_distance_pair(pair_df, metadata, config, log)
            
            if result is None:
                update_stats(stats, "failed")
                continue
            
            update_stats(stats, "successful")
            if result["similarity_score"] > config.get("min_similarity", 0.7):
                update_stats(stats, "high_similarity")
            
            results.append(result)

    # Save results
    if results:
        output_path = save_results(
            results, 
            repo_root / "reports" / "distance",
            f"distance_{interval}.csv",
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
        description="Distance-based analysis for cryptocurrency pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser = add_common_arguments(parser)
    args = parser.parse_args()
    
    # Setup logging
    log = setup_logging("distance", args.verbose)
    
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
            log.info("Distance analysis completed")
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