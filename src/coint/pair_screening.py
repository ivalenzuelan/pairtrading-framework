#!/usr/bin/env python3
"""
Pair screening and selection module for cointegration strategy
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from statistical_tests import validate_cointegration, determine_lead_lag_relationship
from common_analysis import get_valid_pairs, WindowPhase


def screen_cointegrated_pairs(data: pd.DataFrame, config: dict, window: WindowPhase, 
                              log: logging.Logger) -> Tuple[List[dict], List[dict]]:
    """
    Stage 0 & 1: Cointegration screening and Granger causality filtering
    
    Args:
        data: Price data DataFrame
        config: Configuration dictionary
        window: Window phase information
        log: Logger instance
        
    Returns:
        Tuple of (all_results, valid_pairs)
    """
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
            pair_info = process_pair(
                forming_data, symbol1, col1, symbol2, col2, 
                config, window, log
            )
            
            if pair_info:
                results.append(pair_info)
                valid_pairs.append(pair_info)
                log.info(
                    f"Found cointegrated pair: {pair_info['leader']}->{pair_info['follower']} "
                    f"(EG={pair_info['eg_pvalue']:.4f}, std={pair_info['std']:.4f}, "
                    f"data={pair_info['data_points']}/{pair_info.get('total_points', 'N/A')} points)"
                )
        
        except Exception as e:
            log.error(f"Error processing pair {symbol1}/{symbol2}: {e}")
            continue
    
    # Return only top pairs (max per window)
    max_pairs = min(len(valid_pairs), config["max_pairs_per_window"])
    return results, valid_pairs[:max_pairs]


def process_pair(forming_data: pd.DataFrame, symbol1: str, col1: str, symbol2: str, col2: str,
                config: dict, window: WindowPhase, log: logging.Logger) -> Optional[dict]:
    """
    Process a single pair through the screening pipeline
    
    Args:
        forming_data: Forming period data
        symbol1: First symbol name
        col1: First column name
        symbol2: Second symbol name
        col2: Second column name
        config: Configuration dictionary
        window: Window phase information
        log: Logger instance
        
    Returns:
        Dictionary with pair information or None if pair doesn't pass screening
    """
    # Extract slice for this pair
    pair_slice = forming_data[[col1, col2]].copy()
    
    # Calculate data completeness
    total_points = len(pair_slice)
    non_na_points = pair_slice.notna().all(axis=1).sum()
    coverage = non_na_points / total_points if total_points > 0 else 0
    
    # Fill missing values within the forming period
    pair_slice = pair_slice.ffill().bfill()
    
    # Remove any remaining NA
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
        return None
    
    # Log data quality
    log.debug(
        f"Processing {symbol1}/{symbol2}: "
        f"{actual_points}/{total_points} points "
        f"({coverage:.1%} coverage)"
    )
    
    # Get price series
    s1 = pair_df[col1]
    s2 = pair_df[col2]
    
    # Validate cointegration
    coint_result = validate_cointegration(s1, s2, config, log)
    if not coint_result:
        log.debug(f"Skipping {symbol1}/{symbol2}: not cointegrated or spread too stable")
        return None
    
    # Determine lead-lag relationship
    lead_lag = determine_lead_lag_relationship(s1, s2, symbol1, symbol2, config, log)
    if not lead_lag:
        log.debug(f"Skipping {symbol1}/{symbol2}: no clear Granger causality")
        return None
    
    leader, follower = lead_lag
    
    # Compile pair information with safe dictionary access
    try:
        pair_info = {
            "pair": f"{symbol1}/{symbol2}",
            "leader": leader,
            "follower": follower,
            "hedge_ratio": coint_result.get("hedge_ratio", 1.0),
            "intercept": coint_result.get("intercept", 0.0),
            "mean": coint_result.get("mean", 0.0),
            "std": coint_result.get("std", 1.0),
            "s1_vol": coint_result.get("s1_vol", s1.std()),
            "s2_vol": coint_result.get("s2_vol", s2.std()),
            "eg_pvalue": coint_result.get("eg_pvalue", 1.0),
            "johansen": coint_result.get("johansen", False),
            "window_iteration": window.iteration,
            "data_points": actual_points,
            "total_points": total_points,
            "coverage": coverage
        }
    except Exception as e:
        log.error(f"Error compiling pair info for {symbol1}/{symbol2}: {e}")
        return None
    
    return pair_info


def validate_data_quality(pair_df: pd.DataFrame, config: dict, symbol1: str, symbol2: str,
                         log: logging.Logger) -> Tuple[bool, dict]:
    """
    Validate data quality for a pair
    
    Args:
        pair_df: DataFrame with pair data
        config: Configuration dictionary
        symbol1: First symbol name
        symbol2: Second symbol name
        log: Logger instance
        
    Returns:
        Tuple of (is_valid, quality_metrics)
    """
    total_points = len(pair_df)
    actual_points = pair_df.dropna().shape[0]
    coverage = actual_points / total_points if total_points > 0 else 0
    
    min_rows = config.get("min_rows_per_window", 100)
    is_valid = actual_points >= min_rows
    
    quality_metrics = {
        "total_points": total_points,
        "actual_points": actual_points,
        "coverage": coverage,
        "min_rows_required": min_rows,
        "is_valid": is_valid
    }
    
    if not is_valid:
        log.debug(
            f"Data quality check failed for {symbol1}/{symbol2}: "
            f"required: {min_rows}, found: {actual_points} "
            f"(coverage: {coverage:.1%})"
        )
    
    return is_valid, quality_metrics