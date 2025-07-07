#!/usr/bin/env python3
"""
Stochastic Differential Residual (SDR) analysis for pair trading.
Quantifies residual returns after accounting for market exposure.
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
# SDR-SPECIFIC FUNCTIONS
# ----------------------------------------------------------------------
def calculate_sdr_residual(returns_i, returns_j, market_returns, pair_tag="", log=None):
    """
    Calculate SDR (Stochastic Differential Residual) for a pair.
    G_t = R_i,t - R_j,t - Γ * r_m,t
    Returns: (gamma, residual_mean, residual_std, sdr_score, adf_pvalue)
    """
    try:
        # Ensure all series have same length and index
        aligned_data = pd.DataFrame({
            'ret_i': returns_i,
            'ret_j': returns_j,
            'mkt': market_returns
        }).dropna()
        
        if len(aligned_data) < 30:
            if log:
                log.debug(f"Insufficient data for SDR calculation: {len(aligned_data)} points")
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Calculate return spread
        spread_returns = aligned_data['ret_i'] - aligned_data['ret_j']
        
        # CAPM regression: spread_returns = gamma * market_returns + epsilon
        X = sm.add_constant(aligned_data['mkt'])
        model = OLS(spread_returns, X)
        results = model.fit()
        
        # Extract gamma coefficient
        gamma = results.params['mkt']
        gamma_pvalue = results.pvalues['mkt']
        
        # Calculate residuals
        residuals = spread_returns - gamma * aligned_data['mkt']
        
        # Calculate SDR score = |mean(residual)| / std(residual)
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        sdr_score = np.abs(residual_mean) / residual_std if residual_std > 0 else 0
        
        # Check stationarity of residuals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_result = adfuller(residuals.dropna())
            adf_pvalue = adf_result[1]
        
        return gamma, gamma_pvalue, residual_mean, residual_std, sdr_score, adf_pvalue
        
    except Exception as exc:
        if log:
            log.debug(f"SDR calculation failed for {pair_tag}: {exc}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def analyze_sdr_pair(pair_df, btc_data, metadata, config, log):
    """Analyze a single pair using SDR method."""
    col1, col2 = metadata["col1"], metadata["col2"]
    symbol1, symbol2 = metadata["symbol1"], metadata["symbol2"]
    pair_tag = f"{symbol1}-{symbol2} {metadata['window_start'].date()}→{metadata['window_end'].date()}"
    
    # Align BTC data with pair data
    aligned_btc = btc_data.reindex(pair_df.index).ffill().bfill()
    
    # Calculate returns
    pair_returns_i = pair_df[col1].pct_change().dropna()
    pair_returns_j = pair_df[col2].pct_change().dropna()
    btc_returns = aligned_btc.pct_change().dropna()
    
    # Calculate SDR metrics
    gamma, gamma_pvalue, res_mean, res_std, sdr_score, adf_pvalue = calculate_sdr_residual(
        pair_returns_i, pair_returns_j, btc_returns, pair_tag, log
    )
    
    if np.isnan(gamma):
        if log:
            log.debug(f"Skipping pair {pair_tag} - SDR calculation failed")
        return None

    # Calculate common metrics
    common_metrics = calculate_common_metrics(pair_df[col1], pair_df[col2])
    
    # Build result
    result = {
        "method": "sdr",
        "interval": metadata.get("interval", ""),
        "window_start": metadata["window_start"].date().isoformat(),
        "window_end": metadata["window_end"].date().isoformat(),
        "asset1": symbol1,
        "asset2": symbol2,
        "rows": metadata["rows"],
        
        # SDR-specific metrics
        "gamma": round(gamma, 6),
        "gamma_pvalue": round(gamma_pvalue, 6),
        "gamma_significant": gamma_pvalue < 0.05,
        "residual_mean": round(res_mean, 6),
        "residual_std": round(res_std, 6),
        "sdr_score": round(sdr_score, 6),
        "adf_pvalue": round(adf_pvalue, 6),
        "stationary_residual": adf_pvalue < 0.05,
        
        # Similarity score (higher is better)
        "similarity_score": round(sdr_score, 6),
    }
    
    # Add common metrics
    result.update(common_metrics)
    
    return result

# ----------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------------------------------------------------
def analyze_interval(config, repo_root, interval, log):
    """Analyze a single interval using SDR method."""
    log.info(f"▶ Analyzing interval {interval} for SDR pairs...")
    stats = create_analysis_stats()
    
    try:
        # Load all data including BTC
        data = load_interval_data(repo_root, config["cryptos"], interval, log)
        
        # Extract BTC data separately
        btc_data = data['BTC']['close'].copy() if 'BTC' in data else None
        if btc_data is None:
            log.error("BTC data not found in dataset")
            return None
            
    except (FileNotFoundError, ValueError) as e:
        log.error(f"Failed to load data for {interval}: {e}")
        return None

    results = []
    windows = generate_windows(
        config["start_date"], config["end_date"],
        config["window_days"], config["step_days"]
    )
    
    # Get all valid pairs (exclude BTC-BTC)
    valid_pairs = get_valid_pairs(data, config["cryptos"])
    
    # Filter out BTC-BTC pair if exists
    valid_pairs = [pair for pair in valid_pairs 
                  if not (pair[0][0] == 'BTC' and pair[1][0] == 'BTC')]

    for w_start, w_end in windows:
        window_results = []
        
        # Get BTC data for this window
        btc_window = btc_data.loc[w_start:w_end]
        
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
            result = analyze_sdr_pair(pair_df, btc_window, metadata, config, log)
            
            if result is None:
                update_stats(stats, "failed")
                continue
            
            update_stats(stats, "successful")
            window_results.append(result)
        
        # Add ranks within this window
        if window_results:
            # Convert to DataFrame for ranking
            df = pd.DataFrame(window_results)
            
            # Rank by SDR score (higher is better)
            df['sdr_rank'] = df['sdr_score'].rank(ascending=False, pct=True)
            
            # Rank by stationarity (stationary residuals are better)
            df['stationary_rank'] = df['stationary_residual'].rank(ascending=False, pct=True)
            
            # Combined rank
            df['composite_rank'] = (df['sdr_rank'] + df['stationary_rank']).rank(ascending=False, pct=True)
            
            # Update results with ranks
            for _, row in df.iterrows():
                result = row.to_dict()
                result['sdr_rank'] = round(result['sdr_rank'], 4)
                result['stationary_rank'] = round(result['stationary_rank'], 4)
                result['composite_rank'] = round(result['composite_rank'], 4)
                results.append(result)
                
                # Track high SDR pairs
                if result['composite_rank'] > config.get("min_sdr_rank", 0.8):
                    update_stats(stats, "high_sdr")

    # Save results
    if results:
        output_path = save_results(
            results, 
            repo_root / "reports" / "sdr",
            f"sdr_{interval}.csv",
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
        description="SDR (Stochastic Differential Residual) analysis for cryptocurrency pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser = add_common_arguments(parser)
    args = parser.parse_args()
    
    # Setup logging
    log = setup_logging("sdr", args.verbose)
    
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
            log.info("SDR analysis completed")
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