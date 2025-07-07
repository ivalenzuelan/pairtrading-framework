#!/usr/bin/env python3
"""
Hurst Exponent analysis for pair trading.
Identifies mean-reverting spreads with anti-persistent behavior.
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller

# Import common utilities
from common_analysis import (
    setup_logging, load_config, load_interval_data, generate_windows,
    get_valid_pairs, extract_pair_data, preprocess_series, check_data_quality,
    calculate_common_metrics, save_results, create_analysis_stats, 
    update_stats, log_final_stats, add_common_arguments, list_data_files
)

# ----------------------------------------------------------------------
# HURST EXPONENT-SPECIFIC FUNCTIONS
# ----------------------------------------------------------------------
def compute_hurst_exponent(spread, method='rs', log=None):
    """
    Compute Hurst exponent using specified method.
    Available methods: 'rs' (rescaled range), 'dfa' (detrended fluctuation analysis)
    """
    n = len(spread)
    min_chunk_size = 10  # Minimum size for subseries
    
    if n < 50:
        if log:
            log.debug(f"Insufficient data for Hurst calculation: {n} points")
        return np.nan
    
    try:
        if method == 'rs':
            # Rescaled Range (R/S) method
            # Calculate R/S for different chunk sizes
            chunk_sizes = np.logspace(np.log10(min_chunk_size), np.log10(n//2), 20).astype(int)
            chunk_sizes = np.unique(chunk_sizes)
            rs_values = []
            
            for chunk_size in chunk_sizes:
                # Split data into chunks
                chunks = [spread[i:i+chunk_size] for i in range(0, n - chunk_size, chunk_size)]
                if not chunks:
                    continue
                    
                chunk_rs = []
                for chunk in chunks:
                    mean = np.mean(chunk)
                    deviations = chunk - mean
                    cumulative = np.cumsum(deviations)
                    r = np.max(cumulative) - np.min(cumulative)  # Range
                    s = np.std(chunk)  # Standard deviation
                    if s > 0:
                        chunk_rs.append(r / s)
                
                if chunk_rs:
                    rs_values.append(np.mean(chunk_rs))
            
            if len(rs_values) < 5:
                return np.nan
                
            # Fit linear regression: log(R/S) = H * log(n) + c
            x = np.log(np.array(chunk_sizes[:len(rs_values)]))
            y = np.log(np.array(rs_values))
            x = sm.add_constant(x)  # Add constant for intercept
            model = OLS(y, x)
            results = model.fit()
            hurst = results.params[1]
            
            return hurst
            
        elif method == 'dfa':
            # Detrended Fluctuation Analysis
            # Create cumulative sum of deviations
            cumsum = np.cumsum(spread - np.mean(spread))
            
            # Calculate DFA for different window sizes
            window_sizes = np.logspace(np.log10(min_chunk_size), np.log10(n//4), 15).astype(int)
            window_sizes = np.unique(window_sizes)
            f_values = []
            
            for window in window_sizes:
                # Split data into windows
                windows = [cumsum[i:i+window] for i in range(0, n - window, window)]
                if not windows:
                    continue
                    
                window_f = []
                for win in windows:
                    # Detrend by linear regression
                    x = np.arange(len(win))
                    x = sm.add_constant(x)
                    model = OLS(win, x)
                    results = model.fit()
                    trend = results.predict()
                    detrended = win - trend
                    # Calculate RMS fluctuation
                    f = np.sqrt(np.mean(detrended**2))
                    window_f.append(f)
                
                if window_f:
                    f_values.append(np.mean(window_f))
            
            if len(f_values) < 5:
                return np.nan
                
            # Fit linear regression: log(F) = H * log(n) + c
            x = np.log(window_sizes[:len(f_values)])
            y = np.log(f_values)
            x = sm.add_constant(x)
            model = OLS(y, x)
            results = model.fit()
            hurst = results.params[1]
            
            return hurst
            
        else:
            raise ValueError(f"Unknown Hurst method: {method}")
            
    except Exception as e:
        if log:
            log.debug(f"Hurst calculation failed: {e}")
        return np.nan

def analyze_hurst_pair(pair_df, metadata, config, log):
    """Analyze a single pair using Hurst exponent."""
    col1, col2 = metadata["col1"], metadata["col2"]
    symbol1, symbol2 = metadata["symbol1"], metadata["symbol2"]
    pair_tag = f"{symbol1}-{symbol2} {metadata['window_start'].date()}→{metadata['window_end'].date()}"
    
    # Calculate spread
    spread = pair_df[col1] - pair_df[col2]
    
    # Check spread stationarity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_result = adfuller(spread.dropna())
        adf_pvalue = adf_result[1]
    
    # Calculate Hurst exponent using both methods
    hurst_rs = compute_hurst_exponent(spread, 'rs', log)
    hurst_dfa = compute_hurst_exponent(spread, 'dfa', log)
    
    # Use the average if both are valid, otherwise use whichever is available
    if not np.isnan(hurst_rs) and not np.isnan(hurst_dfa):
        hurst = (hurst_rs + hurst_dfa) / 2
        method_used = 'both'
    elif not np.isnan(hurst_rs):
        hurst = hurst_rs
        method_used = 'rs'
    elif not np.isnan(hurst_dfa):
        hurst = hurst_dfa
        method_used = 'dfa'
    else:
        if log:
            log.debug(f"Skipping pair {pair_tag} - Hurst calculation failed")
        return None

    # Calculate common metrics
    common_metrics = calculate_common_metrics(pair_df[col1], pair_df[col2])
    
    # Calculate similarity score (lower H is better for mean-reversion)
    # Scale: 0 = perfect mean-reversion, 1 = strong trending
    similarity_score = max(0, min(1, (0.5 - hurst) * 2)) if hurst < 0.5 else 0
    
    # Build result
    result = {
        "method": "hurst",
        "interval": metadata.get("interval", ""),
        "window_start": metadata["window_start"].date().isoformat(),
        "window_end": metadata["window_end"].date().isoformat(),
        "asset1": symbol1,
        "asset2": symbol2,
        "rows": metadata["rows"],
        
        # Hurst-specific metrics
        "hurst_exponent": round(hurst, 4),
        "hurst_rs": round(hurst_rs, 4) if not np.isnan(hurst_rs) else np.nan,
        "hurst_dfa": round(hurst_dfa, 4) if not np.isnan(hurst_dfa) else np.nan,
        "calculation_method": method_used,
        "adf_pvalue": round(adf_pvalue, 6),
        "stationary_spread": adf_pvalue < 0.05,
        "anti_persistent": hurst < 0.5,
        
        # Similarity score (higher is better for mean-reversion)
        "similarity_score": round(similarity_score, 4),
    }
    
    # Add common metrics
    result.update(common_metrics)
    
    return result

# ----------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------------------------------------------------
def analyze_interval(config, repo_root, interval, log):
    """Analyze a single interval using Hurst exponent."""
    log.info(f"▶ Analyzing interval {interval} for mean-reverting pairs...")
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
            result = analyze_hurst_pair(pair_df, metadata, config, log)
            
            if result is None:
                update_stats(stats, "failed")
                continue
            
            update_stats(stats, "successful")
            window_results.append(result)
        
        # Add ranks within this window
        if window_results:
            # Convert to DataFrame for ranking
            df = pd.DataFrame(window_results)
            
            # Rank by Hurst exponent (lower is better)
            df['hurst_rank'] = df['hurst_exponent'].rank(ascending=True, pct=True)
            
            # Rank by similarity score (higher is better)
            df['similarity_rank'] = df['similarity_score'].rank(ascending=False, pct=True)
            
            # Combined rank
            df['composite_rank'] = (df['hurst_rank'] + df['similarity_rank']).rank(ascending=False, pct=True)
            
            # Update results with ranks
            for _, row in df.iterrows():
                result = row.to_dict()
                result['hurst_rank'] = round(result['hurst_rank'], 4)
                result['similarity_rank'] = round(result['similarity_rank'], 4)
                result['composite_rank'] = round(result['composite_rank'], 4)
                results.append(result)
                
                # Track anti-persistent pairs
                if result['anti_persistent'] and result['composite_rank'] > config.get("min_hurst_rank", 0.8):
                    update_stats(stats, "anti_persistent")

    # Save results
    if results:
        output_path = save_results(
            results, 
            repo_root / "reports" / "hurst",
            f"hurst_{interval}.csv",
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
        description="Hurst Exponent analysis for cryptocurrency pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser = add_common_arguments(parser)
    args = parser.parse_args()
    
    # Setup logging
    log = setup_logging("hurst", args.verbose)
    
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
            log.info("Hurst exponent analysis completed")
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