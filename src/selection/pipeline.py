#!/usr/bin/env python3
"""
Cryptocurrency Pairs Trading Pipeline

Implements a complete backtesting system with:
- 5-day training (forming) period using statistical methods
- 1-day testing (trading) period with up to 5-day buffer zone
- Sliding window approach with daily progression
- Position management and risk controls
- Performance tracking and reporting
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint
from dtaidistance import dtw

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------
# 0 ─── Configuration and Data Structures ─────────────────────────────
# ----------------------------------------------------------------------

class PositionType(Enum):
    LONG_SHORT = "long_short"  # Long asset1, Short asset2
    SHORT_LONG = "short_long"  # Short asset1, Long asset2

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"

@dataclass
class TradingConfig:
    """Trading configuration parameters."""
    # Experimental design parameters
    training_days: int = 5
    testing_days: int = 1
    buffer_days: int = 5
    max_pairs_per_window: int = 5
    initial_capital_per_pair: float = 1000.0  # USD
    initial_capital_per_asset: float = 500.0  # USD per asset in pair
    
    # Trading parameters
    trading_fee: float = 0.0004  # 0.04% as taker fee
    stop_loss_threshold: float = 0.05   # 5% stop loss
    take_profit_threshold: float = 0.02  # 2% take profit
    
    # Statistical thresholds
    cointegration_p_threshold: float = 0.05
    correlation_threshold: float = 0.7
    dtw_threshold: float = 10.0
    
    # Risk management
    max_position_size: float = 0.2  # 20% of capital per position
    min_trading_volume: float = 100.0  # Minimum daily volume

@dataclass
class StatisticalResult:
    """Results from statistical analysis methods."""
    pair: Tuple[str, str]
    method: str
    score: float
    p_value: Optional[float] = None
    is_significant: bool = False
    additional_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class Position:
    """Trading position tracking."""
    pair: Tuple[str, str]
    position_type: PositionType
    entry_date: pd.Timestamp
    entry_prices: Dict[str, float]
    quantities: Dict[str, float]
    capital_allocated: float
    
    # Position management
    status: PositionStatus = PositionStatus.OPEN
    exit_date: Optional[pd.Timestamp] = None
    exit_prices: Optional[Dict[str, float]] = None
    pnl: float = 0.0
    fees_paid: float = 0.0
    
    # Tracking
    max_profit: float = 0.0
    max_drawdown: float = 0.0
    days_held: int = 0
    training_methods: List[str] = field(default_factory=list)

@dataclass
class WindowResult:
    """Results for a single trading window."""
    window_start: pd.Timestamp
    training_end: pd.Timestamp
    testing_start: pd.Timestamp
    testing_end: pd.Timestamp
    
    # Statistical analysis results
    statistical_results: List[StatisticalResult] = field(default_factory=list)
    selected_pairs: List[Tuple[str, str]] = field(default_factory=list)
    selected_pairs_by_method: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)  # NEW FIELD
    
    # Trading results
    positions: List[Position] = field(default_factory=list)
    total_pnl: float = 0.0
    total_fees: float = 0.0
    success_rate: float = 0.0

# ----------------------------------------------------------------------
# 1 ─── Logging Setup ──────────────────────────────────────────────────
# ----------------------------------------------------------------------

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
)
log = logging.getLogger("trading_pipeline")

# ----------------------------------------------------------------------
# 2 ─── Statistical Analysis Methods ───────────────────────────────────
# ----------------------------------------------------------------------

class EnhancedStatisticalAnalyzer:
    """
    Enhanced statistical analyzer implementing multiple pair selection methods
    from academic literature and practical trading applications.
    
    Methods implemented:
    1. Engle-Granger Cointegration (Enhanced)
    2. Pearson Correlation (Enhanced)
    3. Dynamic Time Warping Distance (Enhanced)
    4. Euclidean Distance (From literature)
    5. Fluctuation Behaviour (From Stübinger & Bredthauer, 2017)
    6. Hurst Exponent (Mean reversion detection)
    7. Stochastic Differential Residual (From Do et al., 2006)
    8. Rolling Correlation Stability (Enhanced)
    9. Volatility Clustering Similarity (Enhanced)
    10. Johansen Cointegration Test
    11. Spread Stationarity Test
    12. Price Ratio Statistics
    """
    
    def __init__(self, config):
        self.config = config
        
        # Enhanced thresholds
        self.cointegration_p_threshold = getattr(config, 'cointegration_p_threshold', 0.05)
        self.correlation_threshold = getattr(config, 'correlation_threshold', 0.7)
        self.dtw_threshold = getattr(config, 'dtw_threshold', 10.0)
        self.hurst_threshold = getattr(config, 'hurst_threshold', 0.5)
        self.distance_threshold = getattr(config, 'distance_threshold', 0.1)
        self.min_observations = getattr(config, 'min_observations', 20)
    
    def analyze_pair(self, data: pd.DataFrame, asset1: str, asset2: str) -> List[StatisticalResult]:
        """Run all enhanced statistical methods on a pair."""
        results = []
        
        # Extract clean data for the pair
        pair_data = data[[asset1, asset2]].dropna()
        if len(pair_data) < self.min_observations:
            log.debug(f"Insufficient data for {asset1}-{asset2}: {len(pair_data)} < {self.min_observations}")
            return results
        
        series1, series2 = pair_data[asset1], pair_data[asset2]
        
        # 1. Enhanced Engle-Granger Cointegration
        results.extend(self._engle_granger_cointegration(series1, series2, asset1, asset2))
        
        # 2. Enhanced Pearson Correlation
        results.extend(self._enhanced_correlation(series1, series2, asset1, asset2))
        
        # 3. Enhanced Dynamic Time Warping
        results.extend(self._enhanced_dtw(series1, series2, asset1, asset2))
        
        # 4. Euclidean Distance (From literature)
        results.extend(self._euclidean_distance(series1, series2, asset1, asset2))
        
        # 5. Fluctuation Behaviour (Stübinger & Bredthauer, 2017)
        results.extend(self._fluctuation_behaviour(series1, series2, asset1, asset2))
        
        # 6. Hurst Exponent Analysis
        results.extend(self._hurst_exponent_analysis(series1, series2, asset1, asset2))
        
        # 7. Stochastic Differential Residual
        results.extend(self._stochastic_differential_residual(series1, series2, asset1, asset2))
        
        # 8. Enhanced Rolling Correlation Stability
        results.extend(self._enhanced_correlation_stability(series1, series2, asset1, asset2))
        
        # 9. Enhanced Volatility Clustering
        results.extend(self._enhanced_volatility_clustering(series1, series2, asset1, asset2))
        
        # 10. Johansen Cointegration Test
        results.extend(self._johansen_cointegration(series1, series2, asset1, asset2))
        
        # 11. Spread Stationarity Test
        #results.extend(self._spread_stationarity_test(series1, series2, asset1, asset2))
        
        # 12. Price Ratio Statistics
        #results.extend(self._price_ratio_statistics(series1, series2, asset1, asset2))
        
        return results
    
    def _engle_granger_cointegration(self, series1: pd.Series, series2: pd.Series, 
                                   asset1: str, asset2: str) -> List[StatisticalResult]:
        """Enhanced Engle-Granger cointegration test with both directions."""
        results = []
        
        try:
            # Test both directions
            for direction, (s1, s2, name_suffix) in [
                ("forward", (series1, series2, "")),
                ("reverse", (series2, series1, "_reverse"))
            ]:
                
                # Perform cointegration test
                coint_stat, p_value, crit_values = coint(s1, s2)
                
                # Calculate additional metrics
                # Run OLS regression to get residuals
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                X = s2.values.reshape(-1, 1)
                y = s1.values
                reg.fit(X, y)
                
                residuals = y - reg.predict(X)
                
                # Test residuals for stationarity
                adf_stat, adf_p, *_ = adfuller(residuals)
                
                # Half-life of mean reversion
                half_life = self._calculate_half_life(residuals)
                
                results.append(StatisticalResult(
                    pair=(asset1, asset2),
                    method=f"engle_granger{name_suffix}",
                    score=max(0, 1 - p_value) if not np.isnan(p_value) else 0,
                    p_value=p_value,
                    is_significant=p_value < self.cointegration_p_threshold,
                    additional_metrics={
                        "coint_statistic": coint_stat,
                        "critical_value_1%": crit_values[0],
                        "critical_value_5%": crit_values[1],
                        "critical_value_10%": crit_values[2],
                        "adf_statistic": adf_stat,
                        "adf_p_value": adf_p,
                        "beta_coefficient": reg.coef_[0],
                        "intercept": reg.intercept_,
                        "half_life": half_life,
                        "residual_std": np.std(residuals)
                    }
                ))
                
        except Exception as e:
            log.debug(f"Engle-Granger cointegration failed for {asset1}-{asset2}: {e}")
        
        return results
    
    def _enhanced_correlation(self, series1: pd.Series, series2: pd.Series, 
                            asset1: str, asset2: str) -> List[StatisticalResult]:
        """Enhanced correlation analysis with multiple variants."""
        results = []
        
        try:
            # 1. Pearson correlation (levels)
            pearson_corr = series1.corr(series2)
            
            # 2. Spearman correlation (ranks)
            spearman_corr = series1.corr(series2, method='spearman')
            
            # 3. Kendall's tau
            kendall_corr = series1.corr(series2, method='kendall')
            
            # 4. Returns correlation
            returns1 = series1.pct_change().dropna()
            returns2 = series2.pct_change().dropna()
            returns_corr = returns1.corr(returns2) if len(returns1) > 10 else 0
            
            # 5. Log returns correlation
            log_returns1 = np.log(series1).diff().dropna()
            log_returns2 = np.log(series2).diff().dropna()
            log_returns_corr = log_returns1.corr(log_returns2) if len(log_returns1) > 10 else 0
            
            correlations = [
                ("pearson", pearson_corr),
                ("spearman", spearman_corr),
                ("kendall", kendall_corr),
                ("returns", returns_corr),
                ("log_returns", log_returns_corr)
            ]
            
            for corr_type, corr_value in correlations:
                if not np.isnan(corr_value):
                    results.append(StatisticalResult(
                        pair=(asset1, asset2),
                        method=f"correlation_{corr_type}",
                        score=abs(corr_value),
                        is_significant=abs(corr_value) > self.correlation_threshold,
                        additional_metrics={
                            f"{corr_type}_correlation": corr_value,
                            "correlation_strength": self._categorize_correlation(abs(corr_value))
                        }
                    ))
                    
        except Exception as e:
            log.debug(f"Enhanced correlation failed for {asset1}-{asset2}: {e}")
        
        return results
    
    def _enhanced_dtw(self, series1: pd.Series, series2: pd.Series,
                    asset1: str, asset2: str) -> List[StatisticalResult]:
        """Fixed DTW implementation with pandas compatibility"""
        results = []
        
        # Fixed normalizations with pandas compatibility
        normalizations = [
            ("robust_zscore", lambda s: self._robust_zscore_fixed(s)),
            ("minmax", lambda s: self._minmax_normalize_safe(s)),
            ("log_scale", lambda s: self._log_normalize_safe(s)),
            ("standard_zscore", lambda s: self._standard_zscore_safe(s))
        ]
        
        for norm_name, norm_func in normalizations:
            try:
                norm1 = norm_func(series1)
                norm2 = norm_func(series2)
                
                # Clean and validate
                norm1 = self._clean_series(norm1)
                norm2 = self._clean_series(norm2)
                
                if len(norm1) < 20 or len(norm2) < 20:
                    continue
                
                # Adaptive window size
                window_size = max(15, int(min(len(norm1), len(norm2)) * 0.2))
                
                try:
                    dtw_distance = dtw.distance(norm1.values, norm2.values,
                                            window=window_size,
                                            use_pruning=True)
                except Exception as dtw_error:
                    # Fallback without pruning if it fails
                    dtw_distance = dtw.distance(norm1.values, norm2.values,
                                            window=window_size)
                
                # Improved normalization
                series_length = (len(norm1) + len(norm2)) / 2
                normalized_dist = dtw_distance / series_length if series_length > 0 else 0
                
                # More robust similarity score
                similarity_score = 1 / (1 + 5 * normalized_dist)
                
                # Adaptive significance threshold
                significance_threshold = max(0.01, 0.025 * np.log(len(norm1)))
                is_significant = normalized_dist < significance_threshold
                
                results.append(StatisticalResult(
                    pair=(asset1, asset2),
                    method=f"dtw_{norm_name}",
                    score=similarity_score,
                    is_significant=is_significant,
                    additional_metrics={
                        'dtw_distance': dtw_distance,
                        'normalized_distance': normalized_dist,
                        'series_length': series_length,
                        'window_size': window_size,
                        'correlation': self._safe_correlation(norm1, norm2),
                        'length_ratio': len(norm1) / max(len(norm2), 1),
                        'significance_threshold': significance_threshold
                    }
                ))
                
            except Exception as e:
                log.warning(f"DTW {norm_name} failed for {asset1}-{asset2}: {str(e)}")
                continue
                
        return results

    def _robust_zscore_fixed(self, series: pd.Series) -> pd.Series:
        """Fixed robust z-score using manual MAD calculation"""
        try:
            median_val = series.median()
            
            # Manual MAD calculation (compatible with all pandas versions)
            mad_val = np.median(np.abs(series - median_val))
            
            # Fallback to std if MAD is zero
            if mad_val == 0 or np.isnan(mad_val):
                mad_val = series.std()
                
            # Additional fallback if std is also zero
            if mad_val == 0 or np.isnan(mad_val):
                mad_val = 1.0
                
            return (series - median_val) / mad_val
            
        except Exception as e:
            log.warning(f"Robust z-score failed, using standard z-score: {str(e)}")
            return self._standard_zscore_safe(series)

    def _minmax_normalize_safe(self, series: pd.Series) -> pd.Series:
        """Safe min-max normalization"""
        try:
            min_val = series.min()
            max_val = series.max()
            range_val = max_val - min_val
            
            if range_val == 0 or np.isnan(range_val):
                # Return zeros if no variation
                return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
            
            return (series - min_val) / range_val
            
        except Exception:
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    def _log_normalize_safe(self, series: pd.Series) -> pd.Series:
        """Safe log normalization"""
        try:
            # Handle negative values by shifting
            min_val = series.min()
            offset = abs(min_val) + 1 if min_val <= 0 else 0
            
            adjusted_series = series + offset + 1e-9
            return np.log(adjusted_series)
            
        except Exception:
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    def _standard_zscore_safe(self, series: pd.Series) -> pd.Series:
        """Safe standard z-score normalization"""
        try:
            mean_val = series.mean()
            std_val = series.std()
            
            if std_val == 0 or np.isnan(std_val):
                return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
                
            return (series - mean_val) / std_val
            
        except Exception:
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    def _clean_series(self, series: pd.Series) -> pd.Series:
        """Clean series by removing inf and nan values"""
        try:
            # Replace inf with nan first
            cleaned = series.replace([np.inf, -np.inf], np.nan)
            
            # Drop nan values
            cleaned = cleaned.dropna()
            
            # Ensure we have numeric data
            cleaned = pd.to_numeric(cleaned, errors='coerce').dropna()
            
            return cleaned
            
        except Exception:
            # Return empty series if cleaning fails
            return pd.Series([], dtype=float)

    def _safe_correlation(self, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate correlation with error handling"""
        try:
            # Align series lengths if different
            min_len = min(len(s1), len(s2))
            if min_len < 2:
                return 0.0
                
            s1_aligned = s1.iloc[:min_len]
            s2_aligned = s2.iloc[:min_len]
            
            corr = s1_aligned.corr(s2_aligned)
            return corr if not np.isnan(corr) else 0.0
            
        except Exception:
            return 0.0

    # Alternative original method fix (minimal changes)
    def enhanced_dtw_minimal_fix(self, series1: pd.Series, series2: pd.Series,
                            asset1: str, asset2: str) -> List[StatisticalResult]:
        """Minimal fix for the original DTW implementation"""
        results = []
        
        # Fixed normalizations - just replace .mad() with manual calculation
        normalizations = [
            ("robust_zscore", lambda s: (s - s.median()) / (np.median(np.abs(s - s.median())) + 1e-9)),
            ("minmax", lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)),
            ("log_scale", lambda s: np.log(s + abs(s.min()) + 1 if s.min() <= 0 else np.log(s + 1e-9)))
        ]
        
        for norm_name, norm_func in normalizations:
            try:
                norm1 = norm_func(series1).replace([np.inf, -np.inf], np.nan).dropna()
                norm2 = norm_func(series2).replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(norm1) < 20 or len(norm2) < 20:
                    continue
                    
                # Adaptive window size
                window_size = max(15, int(len(norm1) * 0.2))
                dtw_distance = dtw.distance(norm1.values, norm2.values,
                                        window=window_size,
                                        use_pruning=True)
                
                # Improved normalization
                max_possible = np.sqrt(len(norm1)**2 + len(norm2)**2)
                normalized_dist = dtw_distance / max_possible if max_possible > 0 else 0
                similarity_score = 1 / (1 + 5 * normalized_dist)
                
                # Adaptive significance threshold
                is_significant = normalized_dist < (0.025 * len(norm1))
                
                results.append(StatisticalResult(
                    pair=(asset1, asset2),
                    method=f"dtw_{norm_name}",
                    score=similarity_score,
                    is_significant=is_significant,
                    additional_metrics={
                        'dtw_distance': dtw_distance,
                        'normalized_distance': normalized_dist,
                        'series_correlation': norm1.corr(norm2) if len(norm1) == len(norm2) else 0.0,
                        'length_ratio': len(norm1) / len(norm2),
                        'window_size': window_size
                    }
                ))
                
            except Exception as e:
                log.warning(f"DTW {norm_name} failed for {asset1}-{asset2}: {str(e)}")
                
        return results
    
    def _euclidean_distance(self, series1: pd.Series, series2: pd.Series, 
                          asset1: str, asset2: str) -> List[StatisticalResult]:
        """Euclidean distance method from literature."""
        results = []
        
        try:
            # Normalize both series to [0, 1]
            norm1 = (series1 - series1.min()) / (series1.max() - series1.min())
            norm2 = (series2 - series2.min()) / (series2.max() - series2.min())
            
            # Calculate Euclidean distance
            euclidean_dist = np.sqrt(np.sum((norm1 - norm2) ** 2))
            
            # Normalize by series length
            normalized_dist = euclidean_dist / len(series1)
            
            # Convert to similarity score (lower distance = higher similarity)
            similarity_score = np.exp(-normalized_dist)
            
            results.append(StatisticalResult(
                pair=(asset1, asset2),
                method="euclidean_distance",
                score=similarity_score,
                is_significant=normalized_dist < self.distance_threshold,
                additional_metrics={
                    "euclidean_distance": euclidean_dist,
                    "normalized_distance": normalized_dist,
                    "max_possible_distance": np.sqrt(2 * len(series1)),
                    "distance_percentile": normalized_dist / np.sqrt(2)
                }
            ))
            
        except Exception as e:
            log.debug(f"Euclidean distance failed for {asset1}-{asset2}: {e}")
        
        return results
    
    def _fluctuation_behaviour(self, series1: pd.Series, series2: pd.Series, 
                             asset1: str, asset2: str) -> List[StatisticalResult]:
        """Fluctuation Behaviour method (Stübinger & Bredthauer, 2017)."""
        results = []
        
        try:
            # Calculate spread
            spread = series1 - series2
            
            # Method 1: Standard deviation of spread
            spread_std = spread.std()
            
            # Method 2: Number of zero crossings (mean crossings)
            spread_centered = spread - spread.mean()
            zero_crossings = len(np.where(np.diff(np.signbit(spread_centered)))[0])
            
            # Normalize zero crossings by series length
            normalized_crossings = zero_crossings / len(spread)
            
            # Method 3: Oscillation intensity (combination of both)
            # Higher std and more crossings indicate better mean-reverting behavior
            oscillation_score = (spread_std * normalized_crossings) if spread_std > 0 else 0
            
            # Method 4: Mean reversion speed
            mean_reversion_speed = self._calculate_mean_reversion_speed(spread)
            
            results.extend([
                StatisticalResult(
                    pair=(asset1, asset2),
                    method="fluctuation_volatility",
                    score=min(spread_std / series1.std(), 1.0),  # Normalize by price volatility
                    is_significant=spread_std > 0,
                    additional_metrics={
                        "spread_std": spread_std,
                        "spread_mean": spread.mean(),
                        "spread_coefficient_variation": spread_std / abs(spread.mean()) if spread.mean() != 0 else 0
                    }
                ),
                StatisticalResult(
                    pair=(asset1, asset2),
                    method="fluctuation_crossings",
                    score=min(normalized_crossings * 10, 1.0),  # Scale to [0,1]
                    is_significant=zero_crossings > 5,
                    additional_metrics={
                        "zero_crossings": zero_crossings,
                        "normalized_crossings": normalized_crossings,
                        "crossing_frequency": zero_crossings / (len(spread) / 252) if len(spread) > 252 else zero_crossings  # annualized
                    }
                ),
                StatisticalResult(
                    pair=(asset1, asset2),
                    method="fluctuation_combined",
                    score=min(oscillation_score, 1.0),
                    is_significant=oscillation_score > 0.01,
                    additional_metrics={
                        "oscillation_score": oscillation_score,
                        "mean_reversion_speed": mean_reversion_speed
                    }
                )
            ])
            
        except Exception as e:
            log.debug(f"Fluctuation behaviour failed for {asset1}-{asset2}: {e}")
        
        return results
    
    def _hurst_exponent_analysis(self, series1: pd.Series, series2: pd.Series, 
                               asset1: str, asset2: str) -> List[StatisticalResult]:
        """Hurst Exponent analysis for mean reversion detection."""
        results = []
        
        try:
            # Calculate spread
            spread = series1 - series2
            
            # Calculate Hurst exponent for spread
            spread_hurst = self._calculate_hurst_exponent(spread.values)
            
            # Calculate Hurst exponent for price ratio
            ratio = series1 / series2
            ratio_hurst = self._calculate_hurst_exponent(ratio.values)
            
            # Individual asset Hurst exponents
            asset1_hurst = self._calculate_hurst_exponent(series1.values)
            asset2_hurst = self._calculate_hurst_exponent(series2.values)
            
            # Mean reversion score (lower Hurst = better mean reversion)
            spread_mr_score = max(0, (0.5 - spread_hurst) * 2) if spread_hurst < 0.5 else 0
            ratio_mr_score = max(0, (0.5 - ratio_hurst) * 2) if ratio_hurst < 0.5 else 0
            
            results.extend([
                StatisticalResult(
                    pair=(asset1, asset2),
                    method="hurst_spread",
                    score=spread_mr_score,
                    is_significant=spread_hurst < self.hurst_threshold,
                    additional_metrics={
                        "hurst_exponent": spread_hurst,
                        "mean_reversion_strength": 0.5 - spread_hurst if spread_hurst < 0.5 else 0,
                        "market_behavior": self._classify_hurst_behavior(spread_hurst)
                    }
                ),
                StatisticalResult(
                    pair=(asset1, asset2),
                    method="hurst_ratio",
                    score=ratio_mr_score,
                    is_significant=ratio_hurst < self.hurst_threshold,
                    additional_metrics={
                        "hurst_exponent": ratio_hurst,
                        "mean_reversion_strength": 0.5 - ratio_hurst if ratio_hurst < 0.5 else 0,
                        "individual_hurst_1": asset1_hurst,
                        "individual_hurst_2": asset2_hurst,
                        "hurst_divergence": abs(asset1_hurst - asset2_hurst)
                    }
                )
            ])
            
        except Exception as e:
            log.debug(f"Hurst exponent analysis failed for {asset1}-{asset2}: {e}")
        
        return results
    
    def _stochastic_differential_residual(self, series1: pd.Series, series2: pd.Series, 
                                        asset1: str, asset2: str) -> List[StatisticalResult]:
        """Stochastic Differential Residual method (Do et al., 2006)."""
        results = []
        
        try:
            # Calculate returns
            returns1 = series1.pct_change().dropna()
            returns2 = series2.pct_change().dropna()
            
            if len(returns1) < 10 or len(returns2) < 10:
                return results
            
            # Simple market proxy (equal-weighted average of both assets)
            market_returns = (returns1 + returns2) / 2
            
            # Calculate betas (market exposure)
            beta1 = np.cov(returns1, market_returns)[0, 1] / np.var(market_returns)
            beta2 = np.cov(returns2, market_returns)[0, 1] / np.var(market_returns)
            
            # Calculate residual spread (beta-adjusted)
            residual_spread = returns1 - returns2 - (beta1 - beta2) * market_returns
            
            # Test residual spread for stationarity
            try:
                adf_stat, adf_p, *_ = adfuller(residual_spread.dropna())
                is_stationary = adf_p < 0.05
            except:
                adf_stat, adf_p, is_stationary = 0, 1, False
            
            # Calculate residual volatility
            residual_vol = residual_spread.std()
            
            # Score based on stationarity and low volatility
            stationarity_score = max(0, 1 - adf_p) if not np.isnan(adf_p) else 0
            volatility_score = max(0, 1 - min(residual_vol / 0.1, 1))  # Normalize by 10% threshold
            
            combined_score = (stationarity_score + volatility_score) / 2
            
            results.append(StatisticalResult(
                pair=(asset1, asset2),
                method="stochastic_differential_residual",
                score=combined_score,
                p_value=adf_p,
                is_significant=is_stationary and residual_vol < 0.05,
                additional_metrics={
                    "beta_1": beta1,
                    "beta_2": beta2,
                    "beta_difference": abs(beta1 - beta2),
                    "residual_volatility": residual_vol,
                    "adf_statistic": adf_stat,
                    "adf_p_value": adf_p,
                    "is_stationary": is_stationary,
                    "residual_mean": residual_spread.mean(),
                    "residual_skewness": residual_spread.skew(),
                    "residual_kurtosis": residual_spread.kurtosis()
                }
            ))
            
        except Exception as e:
            log.debug(f"Stochastic differential residual failed for {asset1}-{asset2}: {e}")
        
        return results
    
    def _enhanced_correlation_stability(self, series1: pd.Series, series2: pd.Series, 
                                      asset1: str, asset2: str) -> List[StatisticalResult]:
        """Enhanced rolling correlation stability analysis."""
        results = []
        
        try:
            # Multiple window sizes for rolling correlation
            windows = [10, 20, 30, 60] if len(series1) > 60 else [min(10, len(series1)//2)]
            
            for window in windows:
                if window >= len(series1):
                    continue
                    
                rolling_corr = series1.rolling(window=window).corr(series2).dropna()
                
                if len(rolling_corr) < 5:
                    continue
                
                # Stability metrics
                corr_std = rolling_corr.std()
                corr_mean = rolling_corr.mean()
                corr_stability = 1 - (corr_std / abs(corr_mean)) if corr_mean != 0 else 0
                
                # Trend analysis
                corr_trend = self._calculate_trend_slope(rolling_corr.values)
                
                # Regime changes (number of significant changes in correlation)
                regime_changes = self._count_regime_changes(rolling_corr.values)
                
                results.append(StatisticalResult(
                    pair=(asset1, asset2),
                    method=f"correlation_stability_{window}d",
                    score=max(0, corr_stability),
                    is_significant=corr_stability > 0.8 and abs(corr_mean) > 0.5,
                    additional_metrics={
                        "window_size": window,
                        "rolling_corr_mean": corr_mean,
                        "rolling_corr_std": corr_std,
                        "stability_ratio": corr_stability,
                        "correlation_trend": corr_trend,
                        "regime_changes": regime_changes,
                        "min_correlation": rolling_corr.min(),
                        "max_correlation": rolling_corr.max(),
                        "correlation_range": rolling_corr.max() - rolling_corr.min()
                    }
                ))
                
        except Exception as e:
            log.debug(f"Enhanced correlation stability failed for {asset1}-{asset2}: {e}")
        
        return results
    
    def _enhanced_volatility_clustering(self, series1: pd.Series, series2: pd.Series, 
                                      asset1: str, asset2: str) -> List[StatisticalResult]:
        """Enhanced volatility clustering similarity analysis."""
        results = []
        
        try:
            # Calculate different types of volatility measures
            returns1 = series1.pct_change().dropna()
            returns2 = series2.pct_change().dropna()
            
            if len(returns1) < 20 or len(returns2) < 20:
                return results
            
            # 1. Rolling standard deviation
            vol_windows = [5, 10, 20]
            
            for window in vol_windows:
                if window >= len(returns1):
                    continue
                    
                vol1 = returns1.rolling(window=window).std()
                vol2 = returns2.rolling(window=window).std()
                
                vol_corr = vol1.corr(vol2)
                
                if not np.isnan(vol_corr):
                    results.append(StatisticalResult(
                        pair=(asset1, asset2),
                        method=f"volatility_clustering_{window}d",
                        score=abs(vol_corr),
                        is_significant=abs(vol_corr) > 0.6,
                        additional_metrics={
                            "volatility_correlation": vol_corr,
                            "window_size": window,
                            "vol1_mean": vol1.mean(),
                            "vol2_mean": vol2.mean(),
                            "vol_ratio": vol1.mean() / vol2.mean() if vol2.mean() != 0 else 0
                        }
                    ))
            
            # 2. GARCH-like volatility (absolute returns)
            abs_returns1 = returns1.abs()
            abs_returns2 = returns2.abs()
            
            abs_corr = abs_returns1.corr(abs_returns2)
            
            # 3. Squared returns correlation
            sq_returns1 = returns1 ** 2
            sq_returns2 = returns2 ** 2
            
            sq_corr = sq_returns1.corr(sq_returns2)
            
            # 4. Volatility regime correlation
            vol_regime_corr = self._calculate_volatility_regime_correlation(returns1, returns2)
            
            additional_results = [
                ("volatility_absolute", abs_corr),
                ("volatility_squared", sq_corr),
                ("volatility_regime", vol_regime_corr)
            ]
            
            for method_name, corr_value in additional_results:
                if not np.isnan(corr_value):
                    results.append(StatisticalResult(
                        pair=(asset1, asset2),
                        method=method_name,
                        score=abs(corr_value),
                        is_significant=abs(corr_value) > 0.6,
                        additional_metrics={
                            f"{method_name}_correlation": corr_value
                        }
                    ))
                    
        except Exception as e:
            log.debug(f"Enhanced volatility clustering failed for {asset1}-{asset2}: {e}")
        
        return results
    
    def _johansen_cointegration(self, series1: pd.Series, series2: pd.Series, 
                              asset1: str, asset2: str) -> List[StatisticalResult]:
        """Johansen cointegration test."""
        results = []
        
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            
            # Prepare data matrix
            data_matrix = np.column_stack([series1.values, series2.values])
            
            # Perform Johansen test
            johansen_result = coint_johansen(data_matrix, det_order=0, k_ar_diff=1)
            
            # Extract trace and max eigenvalue statistics
            trace_stat = johansen_result.lr1[0]  # Trace statistic for r=0
            max_eigen_stat = johansen_result.lr2[0]  # Max eigenvalue statistic for r=0
            
            # Critical values (95% confidence)
            trace_crit = johansen_result.cvt[0, 1]  # 95% critical value for trace test
            max_eigen_crit = johansen_result.cvm[0, 1]  # 95% critical value for max eigenvalue test
            
            # Test significance
            trace_significant = trace_stat > trace_crit
            max_eigen_significant = max_eigen_stat > max_eigen_crit
            
            # Cointegrating vector
            cointegrating_vector = johansen_result.evec[:, 0]
            
            results.extend([
                StatisticalResult(
                    pair=(asset1, asset2),
                    method="johansen_trace",
                    score=min(trace_stat / trace_crit, 2.0) if trace_crit > 0 else 0,
                    is_significant=trace_significant,
                    additional_metrics={
                        "trace_statistic": trace_stat,
                        "trace_critical_value": trace_crit,
                        "cointegrating_vector_1": cointegrating_vector[0],
                        "cointegrating_vector_2": cointegrating_vector[1],
                        "eigenvalue": johansen_result.eig[0]
                    }
                ),
                StatisticalResult(
                    pair=(asset1, asset2),
                    method="johansen_max_eigen",
                    score=min(max_eigen_stat / max_eigen_crit, 2.0) if max_eigen_crit > 0 else 0,
                    is_significant=max_eigen_significant,
                    additional_metrics={
                        "max_eigen_statistic": max_eigen_stat,
                        "max_eigen_critical_value": max_eigen_crit,
                        "eigenvalue": johansen_result.eig[0]
                    }
                )
            ])
            
        except Exception as e:
            log.debug(f"Johansen cointegration failed for {asset1}-{asset2}: {e}")
        
        return results
# ----------------------------------------------------------------------
# 3 ─── Pair Selection Algorithm ───────────────────────────────────────
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 3 ─── Pair Selection Algorithm (Modified) ───────────────────────────
# ----------------------------------------------------------------------

class PairSelector:
    """Selects trading pairs based on statistical analysis results."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def select_pairs_by_method(self, results: List[StatisticalResult]) -> Dict[str, List[Tuple[str, str]]]:
        """Select top pairs for each statistical method."""
        # Group results by method
        method_groups = {}
        for result in results:
            if result.method not in method_groups:
                method_groups[result.method] = []
            method_groups[result.method].append(result)
        
        selected_pairs_by_method = {}
        
        # For each method, select top pairs
        for method, method_results in method_groups.items():
            # Sort pairs by score for this method
            method_results.sort(key=lambda x: x.score, reverse=True)
            
            # Select top pairs for this method
            top_pairs = []
            count = 0
            for result in method_results:
                if count >= self.config.max_pairs_per_window:
                    break
                top_pairs.append(result.pair)
                count += 1
            
            selected_pairs_by_method[method] = top_pairs
        
        log.info(f"Selected pairs for {len(method_groups)} methods")
        return selected_pairs_by_method
# ----------------------------------------------------------------------
# 4 ─── Position Management ────────────────────────────────────────────
# ----------------------------------------------------------------------

class PositionManager:
    """Manages trading positions with risk controls and method tracking."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def open_position(self, pair: Tuple[str, str], entry_date: pd.Timestamp, 
                     prices: Dict[str, float], signal_direction: int,
                     training_methods: List[str]) -> Position:
        """Open a new trading position."""
        asset1, asset2 = pair
        
        # Determine position type based on signal
        position_type = PositionType.LONG_SHORT if signal_direction > 0 else PositionType.SHORT_LONG
        
        # Calculate quantities (equal dollar amounts)
        quantity1 = self.config.initial_capital_per_asset / prices[asset1]
        quantity2 = self.config.initial_capital_per_asset / prices[asset2]
        
        # Calculate entry fees
        entry_fees = (self.config.initial_capital_per_asset * 2) * self.config.trading_fee
        
        position = Position(
            pair=pair,
            position_type=position_type,
            entry_date=entry_date,
            entry_prices=prices.copy(),
            quantities={asset1: quantity1, asset2: quantity2},
            capital_allocated=self.config.initial_capital_per_pair,
            fees_paid=entry_fees,
            training_methods=training_methods
        )
        
        log.debug(f"Opened {position_type.value} position for {asset1}-{asset2} at {entry_date}")
        return position
    
    def update_position(self, position: Position, current_date: pd.Timestamp, 
                       current_prices: Dict[str, float]) -> None:
        """Update position metrics and check for exit conditions."""
        if position.status != PositionStatus.OPEN:
            return
        
        asset1, asset2 = position.pair
        
        # Calculate current P&L
        if position.position_type == PositionType.LONG_SHORT:
            # Long asset1, short asset2
            pnl1 = position.quantities[asset1] * (current_prices[asset1] - position.entry_prices[asset1])
            pnl2 = position.quantities[asset2] * (position.entry_prices[asset2] - current_prices[asset2])
        else:
            # Short asset1, long asset2
            pnl1 = position.quantities[asset1] * (position.entry_prices[asset1] - current_prices[asset1])
            pnl2 = position.quantities[asset2] * (current_prices[asset2] - position.entry_prices[asset2])
        
        current_pnl = pnl1 + pnl2
        position.pnl = current_pnl
        
        # Update tracking metrics
        position.max_profit = max(position.max_profit, current_pnl)
        position.max_drawdown = min(position.max_drawdown, current_pnl)
        position.days_held = (current_date - position.entry_date).days
        
        # Check exit conditions
        pnl_percentage = current_pnl / position.capital_allocated
        
        should_exit = False
        exit_reason = ""
        
        # Stop loss
        if pnl_percentage <= -self.config.stop_loss_threshold:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Take profit
        elif pnl_percentage >= self.config.take_profit_threshold:
            should_exit = True
            exit_reason = "take_profit"
        
        # Maximum holding period (testing + buffer)
        elif position.days_held >= (self.config.testing_days + self.config.buffer_days):
            should_exit = True
            exit_reason = "max_holding_period"
        
        if should_exit:
            self.close_position(position, current_date, current_prices, exit_reason)
    
    def close_position(self, position: Position, exit_date: pd.Timestamp, 
                      exit_prices: Dict[str, float], reason: str) -> None:
        """Close a trading position."""
        position.status = PositionStatus.CLOSED
        position.exit_date = exit_date
        position.exit_prices = exit_prices.copy()
        
        # Add exit fees
        exit_fees = (self.config.initial_capital_per_asset * 2) * self.config.trading_fee
        position.fees_paid += exit_fees
        
        # Final P&L calculation (already calculated in update_position)
        final_pnl = position.pnl - position.fees_paid
        position.pnl = final_pnl
        
        log.debug(f"Closed position {position.pair[0]}-{position.pair[1]} on {exit_date} "
                 f"(reason: {reason}, P&L: ${final_pnl:.2f})")

# ----------------------------------------------------------------------
# 5 ─── Main Trading Pipeline ──────────────────────────────────────────
# ----------------------------------------------------------------------

class TradingPipeline:
    """Main trading pipeline orchestrating the entire process."""
    
    def __init__(self, config: TradingConfig, data_path: Path):
        self.config = config
        self.data_path = data_path
        self.analyzer = EnhancedStatisticalAnalyzer(config)
        self.selector = PairSelector(config)
        self.position_manager = PositionManager(config)
        
        # Pipeline state
        self.active_positions: List[Position] = []
        self.window_results: List[WindowResult] = []
    
    def load_data(self, cryptos: List[str], start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """Load cryptocurrency price data from Parquet files."""
        log.info(f"Loading data for {len(cryptos)} cryptocurrencies from {start_date} to {end_date}")
        
        # Define data directory structure (similar to the correlation script)
        proc_dir = self.data_path / "processed" / interval
        
        if not proc_dir.exists():
            log.warning(f"Processed data directory not found: {proc_dir}")
            exit(1)
        
        dfs = []
        missing = []
        
        # Load each cryptocurrency's data
        for crypto in cryptos:
            crypto_lower = crypto.lower()
            parquet_file = proc_dir / f"{crypto_lower}.parquet"
            
            if parquet_file.exists():
                try:
                    # Load the parquet file
                    df = pd.read_parquet(parquet_file)
                    
                    # Extract close price and rename column
                    if "close" in df.columns:
                        crypto_data = df[["close"]].rename(columns={"close": crypto})
                        dfs.append(crypto_data)
                        log.debug(f"Loaded {crypto} data: {len(crypto_data)} rows")
                    else:
                        log.warning(f"No 'close' column found in {parquet_file}")
                        missing.append(crypto)
                        
                except Exception as e:
                    log.warning(f"Failed to load {parquet_file}: {e}")
                    missing.append(crypto)
            else:
                missing.append(crypto)
        
        if missing:
            log.warning(f"Missing {len(missing)}/{len(cryptos)} symbols: {', '.join(missing)}")
        
        if not dfs:
            exit(1)
        
        # Merge all dataframes
        merged_data = pd.concat(dfs, axis=1).sort_index()
        
        # Filter by date range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Filter data to the specified date range
        if merged_data.index.min() <= start_ts and merged_data.index.max() >= end_ts:
            filtered_data = merged_data.loc[start_ts:end_ts]
        else:
            log.warning(f"Data range {merged_data.index.min()} to {merged_data.index.max()} "
                    f"doesn't fully cover requested range {start_ts} to {end_ts}")
            filtered_data = merged_data
        
        log.info(f"Loaded {len(dfs)} symbols, {len(filtered_data)} rows after date filtering")
        
        
        return filtered_data
    
    def generate_trading_windows(self, start_date: str, end_date: str) -> List[WindowResult]:
        """Generate sliding windows for the experimental design."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        windows = []
        current_date = start
        
        while current_date <= end:
            # Define window periods
            training_start = current_date
            training_end = current_date + pd.Timedelta(days=self.config.training_days - 1)
            testing_start = training_end + pd.Timedelta(days=1)
            testing_end = testing_start + pd.Timedelta(days=self.config.testing_days - 1)
            
            # Check if we have enough data for the complete window + buffer
            buffer_end = testing_end + pd.Timedelta(days=self.config.buffer_days)
            if buffer_end > end:
                break
            
            window = WindowResult(
                window_start=training_start,
                training_end=training_end,
                testing_start=testing_start,
                testing_end=buffer_end  # Include buffer period
            )
            windows.append(window)
            
            # Move to next day
            current_date += pd.Timedelta(days=1)
        
        log.info(f"Generated {len(windows)} trading windows")
        return windows
    

    def run_training_phase(self, window: WindowResult, data: pd.DataFrame, 
                          cryptos: List[str]) -> None:
        """Run the 5-day training phase for statistical analysis."""
        log.info(f"Training phase: {window.window_start.date()} to {window.training_end.date()}")
        
        # Extract training data
        training_data = data.loc[window.window_start:window.training_end]
        
        # Run statistical analysis on all pairs
        from itertools import combinations
        all_results = []
        
        for asset1, asset2 in combinations(cryptos, 2):
            if asset1 in training_data.columns and asset2 in training_data.columns:
                pair_results = self.analyzer.analyze_pair(training_data, asset1, asset2)
                all_results.extend(pair_results)
        
        window.statistical_results = all_results
        
        # Select best pairs per method
        window.selected_pairs_by_method = self.selector.select_pairs_by_method(all_results)
        
        # Flatten all selected pairs
        all_selected_pairs = set()
        for pairs in window.selected_pairs_by_method.values():
            all_selected_pairs.update(pairs)
        
        window.selected_pairs = list(all_selected_pairs)[:self.config.max_pairs_per_window]
        
        log.info(f"Training completed: selected {len(window.selected_pairs)} pairs "
                f"from {len(all_results)} pair-method combinations")

    def generate_trading_signal(self, pair_data: pd.DataFrame, asset1: str, asset2: str) -> int:
        """Generate trading signal based on z-score of price ratio (paper-matching)."""
        if len(pair_data) < 5:
            return 0
        
        # Calculate price ratio
        ratio = pair_data[asset1] / pair_data[asset2]
        
        # Calculate z-score of the ratio
        mean_ratio = ratio.mean()
        std_ratio = ratio.std()
        
        # Use the most recent ratio value
        recent_ratio = ratio.iloc[-1]
        
        # Calculate z-score
        z_score = (recent_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
        
        # Generate signal based on z-score thresholds
        if z_score > 1.0:
            return -1  # SHORT_LONG (short asset1, long asset2)
        elif z_score < -1.0:
            return 1   # LONG_SHORT (long asset1, short asset2)
        else:
            return 0   # No signal
    
    def run_trading_phase(self, window: WindowResult, data: pd.DataFrame) -> None:
        """Run the trading phase with position management."""
        log.info(f"Trading phase: {window.testing_start.date()} to {window.testing_end.date()}")
        
        # Open positions for selected pairs
        current_date = window.testing_start
        
        if current_date in data.index:
            current_prices = data.loc[current_date].to_dict()
            
            for pair in window.selected_pairs:
                asset1, asset2 = pair

                # Find which methods selected this pair
                selection_methods = []
                for method, pairs in window.selected_pairs_by_method.items():
                    if pair in pairs:
                        selection_methods.append(method)
                
                # Generate trading signal based on training data
                training_data = data.loc[window.window_start:window.training_end]
                signal = self.generate_trading_signal(training_data, asset1, asset2)
                
                if signal != 0:  # Only trade if we have a signal
                    try:
                        position = self.position_manager.open_position(
                            pair, current_date, current_prices, signal, 
                            training_methods=selection_methods
                        )
                        window.positions.append(position)
                        self.active_positions.append(position)
                    except Exception as e:
                        log.warning(f"Failed to open position for {asset1}-{asset2}: {e}")
        
        # Update positions daily during trading and buffer period
        trading_dates = pd.date_range(
            start=window.testing_start, 
            end=window.testing_end, 
            freq='D'
        )
        
        for date in trading_dates:
            if date in data.index:
                current_prices = data.loc[date].to_dict()
                
                # Update all active positions for this window
                for position in window.positions:
                    if position.status == PositionStatus.OPEN:
                        self.position_manager.update_position(position, date, current_prices)
        
        # Close any remaining open positions at the end of buffer period
        final_date = window.testing_end
        if final_date in data.index:
            final_prices = data.loc[final_date].to_dict()
            for position in window.positions:
                if position.status == PositionStatus.OPEN:
                    self.position_manager.close_position(position, final_date, final_prices, "buffer_end")
        
        # Calculate window performance
        window.total_pnl = sum(pos.pnl for pos in window.positions)
        window.total_fees = sum(pos.fees_paid for pos in window.positions)
        profitable_positions = [pos for pos in window.positions if pos.pnl > 0]
        window.success_rate = len(profitable_positions) / len(window.positions) if window.positions else 0
        
        log.info(f"Trading completed: {len(window.positions)} positions, "
                f"P&L: ${window.total_pnl:.2f}, Success rate: {window.success_rate:.2%}")
    
    def run_backtest(self, cryptos: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """Run the complete backtesting pipeline."""
        log.info(f"Starting backtest from {start_date} to {end_date}")
        start_time = time.time()
        
        # Load data
        data = self.load_data(cryptos, start_date, end_date)
        
        # Generate trading windows
        windows = self.generate_trading_windows(start_date, end_date)
        
        # Process each window
        for i, window in enumerate(windows):
            log.info(f"Processing window {i+1}/{len(windows)}: {window.window_start.date()}")
            
            try:
                # Training phase
                self.run_training_phase(window, data, cryptos)
                
                # Trading phase
                self.run_trading_phase(window, data)
                
                self.window_results.append(window)
                
            except Exception as e:
                log.error(f"Failed to process window {i+1}: {e}")
                continue
        
        # Generate final results
        results = self.generate_results_summary()
        
        processing_time = time.time() - start_time
        log.info(f"Backtest completed in {processing_time:.2f}s")
        
        return results
    
    def generate_results_summary(self) -> Dict[str, Any]:
        """Generate comprehensive results summary."""
        if not self.window_results:
            return {"error": "No results to summarize"}
        
        # Aggregate metrics
        total_windows = len(self.window_results)
        total_positions = sum(len(w.positions) for w in self.window_results)
        total_pnl = sum(w.total_pnl for w in self.window_results)
        total_fees = sum(w.total_fees for w in self.window_results)
        
        # Performance metrics
        profitable_windows = [w for w in self.window_results if w.total_pnl > 0]
        win_rate = len(profitable_windows) / total_windows if total_windows > 0 else 0
        
        avg_pnl_per_window = total_pnl / total_windows if total_windows > 0 else 0
        avg_positions_per_window = total_positions / total_windows if total_windows > 0 else 0
        
        # All positions for detailed analysis
        all_positions = []
        for window in self.window_results:
            all_positions.extend(window.positions)
        
        profitable_positions = [p for p in all_positions if p.pnl > 0]
        position_success_rate = len(profitable_positions) / len(all_positions) if all_positions else 0
        
        # Risk metrics
        pnl_series = [w.total_pnl for w in self.window_results]
        max_drawdown = min(pnl_series) if pnl_series else 0
        max_profit = max(pnl_series) if pnl_series else 0
        
        # Return performance metrics
        initial_capital = self.config.initial_capital_per_pair * self.config.max_pairs_per_window
        total_return = total_pnl / (initial_capital * total_windows) if total_windows > 0 else 0
        
        # Sharpe ratio approximation
        if len(pnl_series) > 1:
            pnl_std = np.std(pnl_series)
            sharpe_ratio = (avg_pnl_per_window / pnl_std) if pnl_std > 0 else 0
        else:
            sharpe_ratio = 0

        method_performance = {}
        for window in self.window_results:
            for position in window.positions:
                for method in position.training_methods:
                    if method not in method_performance:
                        method_performance[method] = {
                            "total_pnl": 0.0,
                            "total_fees": 0.0,
                            "positions": 0,
                            "winning_positions": 0,
                            "returns": []
                        }
                    
                    method_performance[method]["total_pnl"] += position.pnl
                    method_performance[method]["total_fees"] += position.fees_paid
                    method_performance[method]["positions"] += 1
                    method_performance[method]["returns"].append(position.pnl)
                    
                    if position.pnl > 0:
                        method_performance[method]["winning_positions"] += 1
        
        # Calculate metrics per method
        for method, data in method_performance.items():
            data["avg_return_per_trade"] = data["total_pnl"] / data["positions"] if data["positions"] > 0 else 0
            data["win_rate"] = data["winning_positions"] / data["positions"] if data["positions"] > 0 else 0
        
        # Statistical testing (pairwise method comparison)
        method_comparison = {}
        methods = list(method_performance.keys())
        
        if len(methods) >= 2:
            from itertools import combinations
            from scipy import stats
            
            for method1, method2 in combinations(methods, 2):
                returns1 = method_performance[method1]["returns"]
                returns2 = method_performance[method2]["returns"]
                
                if len(returns1) >= 2 and len(returns2) >= 2:
                    t_stat, p_value = stats.ttest_ind(returns1, returns2, equal_var=False)
                    
                    method_comparison[f"{method1}_vs_{method2}"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant_1%": p_value < 0.01,
                        "significant_5%": p_value < 0.05,
                        "significant_10%": p_value < 0.10
                    }
        
        return {
            "summary": {
                "total_windows": total_windows,
                "total_positions": total_positions,
                "total_pnl": round(total_pnl, 2),
                "total_fees": round(total_fees, 2),
                "net_pnl": round(total_pnl - total_fees, 2),
                "total_return": round(total_return, 4),
                "win_rate": round(win_rate, 4),
                "position_success_rate": round(position_success_rate, 4),
                "avg_pnl_per_window": round(avg_pnl_per_window, 2),
                "avg_positions_per_window": round(avg_positions_per_window, 2),
                "max_profit": round(max_profit, 2),
                "max_drawdown": round(max_drawdown, 2),
                "sharpe_ratio": round(sharpe_ratio, 4)
            },
            "method_performance": method_performance,
            "method_comparison": method_comparison,
            "detailed_results": {
                "window_results": [
                    {
                        "window_start": w.window_start.strftime("%Y-%m-%d"),
                        "pnl": round(w.total_pnl, 2),
                        "fees": round(w.total_fees, 2),
                        "positions": len(w.positions),
                        "success_rate": round(w.success_rate, 4),
                        "selected_pairs": [f"{p[0]}-{p[1]}" for p in w.selected_pairs]
                    }
                    for w in self.window_results
                ],
                "position_details": [
                    {
                        "pair": f"{p.pair[0]}-{p.pair[1]}",
                        "position_type": p.position_type.value,
                        "entry_date": p.entry_date.strftime("%Y-%m-%d"),
                        "exit_date": p.exit_date.strftime("%Y-%m-%d") if p.exit_date else None,
                        "pnl": round(p.pnl, 2),
                        "fees": round(p.fees_paid, 2),
                        "days_held": p.days_held,
                        "status": p.status.value,
                        "max_profit": round(p.max_profit, 2),
                        "max_drawdown": round(p.max_drawdown, 2),
                        "training_methods": p.training_methods
                    }
                    for p in all_positions
                ]
            },
            "config": {
                "training_days": self.config.training_days,
                "testing_days": self.config.testing_days,
                "buffer_days": self.config.buffer_days,
                "max_pairs_per_window": self.config.max_pairs_per_window,
                "initial_capital_per_pair": self.config.initial_capital_per_pair,
                "trading_fee": self.config.trading_fee,
                "stop_loss_threshold": self.config.stop_loss_threshold,
                "take_profit_threshold": self.config.take_profit_threshold
            }
        }

    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        log.info(f"Results saved to {output_path}")

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of results."""
        summary = results.get("summary", {})
        
        print("\n" + "="*60)
        print("CRYPTOCURRENCY PAIRS TRADING BACKTEST RESULTS")
        print("="*60)
        
        print(f"Total Trading Windows: {summary.get('total_windows', 0)}")
        print(f"Total Positions: {summary.get('total_positions', 0)}")
        print(f"Average Positions per Window: {summary.get('avg_positions_per_window', 0)}")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"Total P&L: ${summary.get('total_pnl', 0):,.2f}")
        print(f"Total Fees: ${summary.get('total_fees', 0):,.2f}")
        print(f"Net P&L: ${summary.get('net_pnl', 0):,.2f}")
        print(f"Total Return: {summary.get('total_return', 0):.2%}")
        
        print(f"\nSUCCESS METRICS:")
        print(f"Window Win Rate: {summary.get('win_rate', 0):.2%}")
        print(f"Position Success Rate: {summary.get('position_success_rate', 0):.2%}")
        
        print(f"\nRISK METRICS:")
        print(f"Average P&L per Window: ${summary.get('avg_pnl_per_window', 0):,.2f}")
        print(f"Maximum Profit: ${summary.get('max_profit', 0):,.2f}")
        print(f"Maximum Drawdown: ${summary.get('max_drawdown', 0):,.2f}")
        print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.4f}")
        
        print("="*60)


# ----------------------------------------------------------------------
# 6 ─── Main Execution Function ────────────────────────────────────────
# ----------------------------------------------------------------------

def main():
    """Main execution function."""
    # Configuration
    config = TradingConfig(
        training_days=5,
        testing_days=1,
        buffer_days=5,
        max_pairs_per_window=5,
        initial_capital_per_pair=1000.0,
        trading_fee=0.0004,
        stop_loss_threshold=0.05,
        take_profit_threshold=0.02
    )
    
    # Cryptocurrency list (top 30 by market cap)
    cryptos = [
      "BTCUSDT",
      "ETHUSDT",
      "ADAUSDT",
      "BNBUSDT",
      "SOLUSDT",
      "XRPUSDT",
      "DOGEUSDT",
      "AVAXUSDT" 
    ]
    
    # Trading period
    start_date = "2024-01-01"
    end_date = "2024-01-31"  # One month for demonstration
    
    # Initialize pipeline
    pipeline = TradingPipeline(config, Path("./data"))
    
    try:
        # Run backtest
        log.info("Initializing cryptocurrency pairs trading backtest...")
        results = pipeline.run_backtest(cryptos, start_date, end_date)
        
        # Print results
        pipeline.print_summary(results)
        
        # Save results
        output_path = Path("./results/backtest_results.json")
        pipeline.save_results(results, output_path)
        
        # Optional: Save detailed position data to CSV
        if results.get("detailed_results", {}).get("position_details"):
            positions_df = pd.DataFrame(results["detailed_results"]["position_details"])
            positions_df.to_csv("./results/positions.csv", index=False)
            log.info("Position details saved to ./results/positions.csv")
        
        return results
        
    except Exception as e:
        log.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    # Set up results directory
    Path("./results").mkdir(exist_ok=True)
    
    # Run the main function
    results = main()