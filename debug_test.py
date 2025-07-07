#!/usr/bin/env python3
"""
Debug script to understand why validate_cointegration returns None
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'coint'))

import pandas as pd
import numpy as np
import logging
from statistical_tests import CointegrationAnalyzer

def debug_validate_cointegration():
    """Debug the validate_cointegration method"""
    
    # Setup detailed logging
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)
    
    # Create sample data
    np.random.seed(42)
    n_obs = 200
    
    # Generate cointegrated series
    error_term = np.random.normal(0, 0.1, n_obs)
    series1 = np.cumsum(np.random.normal(0, 1, n_obs)) + error_term
    series2 = 0.8 * series1 + np.cumsum(np.random.normal(0, 0.5, n_obs)) + error_term
    
    s1 = pd.Series(series1, name='MATICUSDT')
    s2 = pd.Series(series2, name='DOGEUSDT')
    
    print(f"Series 1 stats: mean={s1.mean():.4f}, std={s1.std():.4f}")
    print(f"Series 2 stats: mean={s2.mean():.4f}, std={s2.std():.4f}")
    
    # Initialize analyzer
    config = {
        'vecm': {
            'max_lags': 5,
            'coint_rank': 1,
            'deterministic': 'ci'
        }
    }
    
    analyzer = CointegrationAnalyzer(config, log)
    
    # Test individual components
    print("\n=== Testing individual components ===")
    
    # Test _validate_series
    print("Testing _validate_series...")
    is_valid = analyzer._validate_series(s1, s2)
    print(f"_validate_series result: {is_valid}")
    
    # Test Engle-Granger test
    print("\nTesting Engle-Granger test...")
    eg_result = analyzer.engle_granger_test(s1, s2)
    print(f"Engle-Granger result: {eg_result}")
    
    # Test Johansen test
    print("\nTesting Johansen test...")
    aligned = pd.concat([s1, s2], axis=1).dropna()
    johansen_result = analyzer.johansen_test(aligned)
    print(f"Johansen result: {johansen_result}")
    
    # Test spread components
    print("\nTesting spread components...")
    hedge_ratio, intercept, spread_stats = analyzer._calculate_spread_components(s1, s2)
    print(f"Hedge ratio: {hedge_ratio}")
    print(f"Intercept: {intercept}")
    print(f"Spread stats: {spread_stats}")
    
    # Test full validate_cointegration
    print("\n=== Testing full validate_cointegration ===")
    result = analyzer.validate_cointegration(s1, s2)
    print(f"Final result: {result}")

if __name__ == "__main__":
    debug_validate_cointegration() 