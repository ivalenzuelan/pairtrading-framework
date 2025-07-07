#!/usr/bin/env python3
"""
Fluctuation Behavior (FB) analysis for pair trading.
Measures spread volatility and mean-reversion characteristics.
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
# FLUCTUATION BEHAVIOR-SPECIFIC FUNCTIONS
# ----------------------------------------------------------------------
def calculate_spread_metrics(s1, s2, pair_tag="", log=None):
    """
    Calculate fluctuation behavior metrics for a pair.
    Returns: (std_dev, zero_crossings, hurst_exponent, adf_pvalue)
    """
    try:
        # Calculate spread
        spread = s1 - s2
        
        # Standard deviation of spread
        std_dev = spread.std()
        
        # Zero crossings count
        sign_changes = np.diff(np.sign(spread))
        zero_crossings = np.sum(sign_changes != 0)
        
        # Hurst exponent approximation (using rescaled range)
        n = len(spread)
        max_range = np.max(spread.cumsum()) - np.min(spread.cumsum())
        std_dev_spread = spread.std()
        hurst = 0.5 if std_dev_spread == 0 else np.log(max_range / std_dev_spread) / np.log(n)
        
        # ADF test for stationarity
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_result = adfuller(spread.dropna())
            adf_pvalue = adf_result[1]
        
        return std_dev, zero_crossings, hurst, adf_pvalue
        
    except Exception as exc:
        if log:
            log.debug(f"FB metrics failed for {pair_tag}: {exc}")
        return np.nan, np.nan, np.nan, np.nan

def analyze_fluctuation_pair(pair_df, metadata, config, log):
    """Analyze a single pair for fluctuation behavior."""
    col1, col2 = metadata["col1"], metadata["col2"]
    symbol1, symbol2 = metadata["symbol1"], metadata["symbol2"]
    pair_tag = f"{symbol1}-{symbol2} {metadata['window_start'].date()}→{metadata['window_end'].date()}"
    
    # Calculate spread metrics
    std_dev, zero_crossings, hurst, adf_pvalue = calculate_spread_metrics(
        pair_df[col1], pair_df[col2], pair_tag, log
    )
    
    if np.isnan(std_dev):
        if log:
            log.debug(f"Skipping pair {pair_tag} - FB calculation failed")
        return None

    # Calculate common metrics
    common_metrics = calculate_common_metrics(pair_df[col1], pair_df[col2])
    
    # Calculate composite score (higher is better)
    # Rank volatility and mean-reversion separately
    volatility_rank = std_dev  # Higher volatility is better
    mean_reversion_rank = zero_crossings / len(pair_df)  # More crossings = better
    
    # Combine ranks (weighted equally)
    composite_score = 0.5 * volatility_rank + 0.5 * mean_reversion_rank
    
    # Build result
    result = {
        "method": "fluctuation",
        "interval": metadata.get("interval", ""),
        "window_start": metadata["window_start"].date().isoformat(),
        "window_end": metadata["window_end"].date().isoformat(),
        "asset1": symbol1,
        "asset2": symbol2,
        "rows": metadata["rows"],
        
        # FB-specific metrics
        "spread_std": round(std_dev, 6),
        "zero_crossings": int(zero_crossings),
        "zero_crossing_rate": round(zero_crossings / len(pair_df), 4),
        "hurst_exponent": round(hurst, 4),
        "adf_pvalue": round(adf_pvalue, 6),
        "stationary_spread": adf_pvalue < 0.05,
        
        # Composite score
        "volatility_rank": round(volatility_rank, 4),
        "mean_reversion_rank": round(mean_reversion_rank, 4),
        "composite_score": round(composite_score, 4),
        
        # Similarity score (based on composite score)
        "similarity_score": round(composite_score, 4),
    }
    
    # Add common metrics
    result.update(common_metrics)
    
    return result

# ----------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------------------------------------------------
def analyze_interval(config, repo_root, interval, log):
    """Analyze a single interval for fluctuation behavior pairs."""
    log.info(f"▶ Analyzing interval {interval} for fluctuation behavior...")
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
        window_results = []
        
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
            result = analyze_fluctuation_pair(pair_df, metadata, config, log)
            
            if result is None:
                update_stats(stats, "failed")
                continue
            
            update_stats(stats, "successful")
            window_results.append(result)
        
        # Add composite ranks within this window
        if window_results:
            # Convert to DataFrame for ranking
            df = pd.DataFrame(window_results)
            
            # Rank volatility (higher is better)
            df['volatility_rank'] = df['spread_std'].rank(ascending=False, pct=True)
            
            # Rank mean reversion (higher is better)
            df['mean_reversion_rank'] = df['zero_crossing_rate'].rank(ascending=False, pct=True)
            
            # Combined rank
            df['composite_rank'] = (df['volatility_rank'] + df['mean_reversion_rank']).rank(ascending=False, pct=True)
            
            # Update results with ranks
            for _, row in df.iterrows():
                result = row.to_dict()
                result['volatility_rank'] = round(result['volatility_rank'], 4)
                result['mean_reversion_rank'] = round(result['mean_reversion_rank'], 4)
                result['composite_rank'] = round(result['composite_rank'], 4)
                results.append(result)
                
                # Track high FB pairs
                if result['composite_rank'] > config.get("min_fb_rank", 0.8):
                    update_stats(stats, "high_fb")

    # Save results
    if results:
        output_path = save_results(
            results, 
            repo_root / "reports" / "fluctuation",
            f"fluctuation_{interval}.csv",
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
        description="Fluctuation Behavior analysis for cryptocurrency pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser = add_common_arguments(parser)
    args = parser.parse_args()
    
    # Setup logging
    log = setup_logging("fluctuation", args.verbose)
    
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
            log.info("Fluctuation behavior analysis completed")
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