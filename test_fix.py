#!/usr/bin/env python3
"""
Test script to verify the validate_cointegration method fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'coint'))

import pandas as pd
import numpy as np
import logging
from statistical_tests import CointegrationAnalyzer

def test_validate_cointegration():
    """Test the validate_cointegration method"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    
    # Create sample data with proper cointegration
    np.random.seed(42)
    n_obs = 200
    
    # Generate properly cointegrated series
    # Start with a random walk
    random_walk = np.cumsum(np.random.normal(0, 1, n_obs))
    
    # Create two series that are cointegrated
    # s1 = random walk
    # s2 = 0.8 * s1 + stationary error
    s1 = pd.Series(random_walk, name='MATICUSDT')
    s2 = pd.Series(0.8 * random_walk + np.random.normal(0, 0.5, n_obs), name='DOGEUSDT')
    
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
    
    try:
        # Test the validate_cointegration method
        result = analyzer.validate_cointegration(s1, s2)
        
        if result is not None:
            print("✅ validate_cointegration method works!")
            print(f"Result keys: {list(result.keys())}")
            print(f"Cointegrated: {result.get('cointegrated', False)}")
            print(f"Hedge ratio: {result.get('hedge_ratio', 'N/A')}")
            print(f"EG p-value: {result.get('eg_pvalue', 'N/A')}")
            return True
        else:
            print("❌ validate_cointegration returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error in validate_cointegration: {e}")
        return False

def test_determine_lead_lag():
    """Test the determine_lead_lag_relationship method"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    
    # Create sample data
    np.random.seed(42)
    n_obs = 200
    
    # Generate series with lead-lag relationship
    series1 = np.cumsum(np.random.normal(0, 1, n_obs))
    series2 = 0.8 * np.roll(series1, -2) + np.random.normal(0, 0.5, n_obs)  # series1 leads by 2 periods
    
    s1 = pd.Series(series1, name='MATICUSDT')
    s2 = pd.Series(series2, name='DOGEUSDT')
    
    # Initialize analyzer
    config = {}
    analyzer = CointegrationAnalyzer(config, log)
    
    try:
        # Test the determine_lead_lag_relationship method
        result = analyzer.determine_lead_lag_relationship(s1, s2, 'MATICUSDT', 'DOGEUSDT')
        
        if result is not None:
            print("✅ determine_lead_lag_relationship method works!")
            print(f"Leader: {result.get('leader', 'N/A')}")
            print(f"Follower: {result.get('follower', 'N/A')}")
            print(f"Lag periods: {result.get('lag_periods', 'N/A')}")
            return True
        else:
            print("❌ determine_lead_lag_relationship returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error in determine_lead_lag_relationship: {e}")
        return False

if __name__ == "__main__":
    print("Testing validate_cointegration method fix...")
    
    test1_passed = test_validate_cointegration()
    test2_passed = test_determine_lead_lag()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The fix is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 