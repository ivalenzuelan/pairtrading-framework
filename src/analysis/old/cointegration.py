#!/usr/bin/env python3
"""
Cointegration analysis using common utilities.
Simplified version that focuses only on cointegration-specific logic.
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tools.sm_exceptions import CollinearityWarning

# Import common utilities
from common_analysis import (
    setup_logging, load_config, load_interval_data, generate_windows,
    get_valid_pairs, extract_pair_data, preprocess_series, check_data_quality,
    calculate_common_metrics, save_results, create_analysis_stats, 
    update_stats, log_final_stats, add_common_arguments, list_data_files
)

# ----------------------------------------------------------------------
# COINTEGRATION-SPECIFIC FUNCTIONS
# ----------------------------------------------------------------------
def engle_granger_robust(s1, s2, pair_tag="", preprocess_method='log_prices', log=None):
    """Robust Engle-Granger test with preprocessing and error handling."""
    try:
        # Preprocess data
        s1_proc, s2_proc = preprocess_series(s1, s2, preprocess_method)
        
        # Check data quality
        is_valid, issues = check_data_quality(s1_proc, s2_proc, pair_tag)
        if not is_valid:
            if log:
                log.debug(f"Data quality issues for {pair_tag}: {', '.join(issues)}")
            return np.nan
        
        # Ensure we have enough data after preprocessing
        if len(s1_proc) < 30:
            if log:
                log.debug(f"Insufficient data after preprocessing for {pair_tag}: {len(s1_proc)} points")
            return np.nan
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, pvalue, _ = coint(s1_proc, s2_proc)
            
            # Sanity check the result
            if np.isnan(pvalue) or np.isinf(pvalue):
                return np.nan
                
            return pvalue
            
    except Exception as exc:
        if log:
            log.debug(f"EG failed for {pair_tag}: {exc}")
        return np.nan

def johansen_trace_robust(df, pair_tag="", preprocess_method='log_prices', log=None):
    """Robust Johansen test with preprocessing and error handling."""
    try:
        # Preprocess data
        if preprocess_method == 'log_prices':
            df_proc = np.log(df)
        elif preprocess_method == 'standardize':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df_proc = pd.DataFrame(
                scaler.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
        else:
            df_proc = df
        
        # Check for data quality issues
        if df_proc.isnull().any().any() or np.isinf(df_proc.values).any():
            return False
        
        # Check condition number (measure of multicollinearity)
        cond_num = np.linalg.cond(df_proc.cov())
        if cond_num > 1e12:  # Very high condition number
            if log:
                log.debug(f"High condition number for {pair_tag}: {cond_num:.2e}")
            return False
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            j = coint_johansen(df_proc, det_order=-1, k_ar_diff=1)
            return bool(j.lr1[0] > j.cvt[0, 1])     # 95% confidence level
            
    except Exception as exc:
        if log:
            log.debug(f"Johansen failed for {pair_tag}: {exc}")
        return False

def analyze_cointegration_pair(pair_df, metadata, config, log):
    """Analyze a single pair for cointegration."""
    col1, col2 = metadata["col1"], metadata["col2"]
    tag = metadata["tag"]
    
    # Use robust methods with different preprocessing approaches
    p_log = engle_granger_robust(pair_df[col1], pair_df[col2], tag, 'log_prices', log)
    j_log = johansen_trace_robust(pair_df, tag, 'log_prices', log)
    
    # If log prices fail, try standardization
    if np.isnan(p_log):
        p_std = engle_granger_robust(pair_df[col1], pair_df[col2], tag, 'standardize', log)
        j_std = johansen_trace_robust(pair_df, tag, 'standardize', log)
        p, j = p_std, j_std
        method_used = 'standardized'
    else:
        p, j = p_log, j_log
        method_used = 'log_prices'

    if np.isnan(p):
        return None

    # Calculate common metrics
    common_metrics = calculate_common_metrics(pair_df[col1], pair_df[col2])
    
    # Determine if cointegrated
    is_cointegrated = (p < config["p_threshold"]) or j
    
    # Build result
    result = {
        "method": "cointegration",
        "interval": metadata.get("interval", ""),
        "window_start": metadata["window_start"].date().isoformat(),
        "window_end": metadata["window_end"].date().isoformat(),
        "asset1": metadata["symbol1"],
        "asset2": metadata["symbol2"],
        "rows": metadata["rows"],
        
        # Cointegration-specific metrics
        "eg_pvalue": round(p, 6),
        "johansen_95": j,
        "cointegrated": is_cointegrated,
        "preprocessing_method": method_used,
        
        # Similarity score (inverse of p-value for comparison)
        "similarity_score": round(1 - p if p < 1 else 0, 6),
    }
    
    # Add common metrics
    result.update(common_metrics)
    
    return result

# ----------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------------------------------------------------
def analyze_interval(config, repo_root, interval, log):
    """Analyze a single interval for cointegration."""
    log.info(f"▶ Analyzing interval {interval}...")
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
            result = analyze_cointegration_pair(pair_df, metadata, config, log)
            
            if result is None:
                update_stats(stats, "failed")
                continue
            
            update_stats(stats, "successful")
            if result["cointegrated"]:
                # We can track cointegrated pairs separately if needed
                pass
            
            results.append(result)

    # Save results
    if results:
        output_path = save_results(
            results, 
            repo_root / "reports" / "cointegration",
            f"cointegration_{interval}.csv",
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
        description="Cointegration analysis for cryptocurrency pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser = add_common_arguments(parser)
    args = parser.parse_args()
    
    # Setup logging
    log = setup_logging("cointegration", args.verbose)
    
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
            log.info("Analysis completed successfully")
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