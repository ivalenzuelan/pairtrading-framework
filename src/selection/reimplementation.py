#!/usr/bin/env python3
"""
Cryptocurrency Pairs Trading Pipeline - Paper Reimplementation

Implements the experimental design from:
"An Empirical Analysis of Cryptocurrency Pairs Trading: 
Exploring Statistical Methods for Pair Selection"

Key features:
- Supports 1d, 15m, 1h frequencies
- Six statistical methods from paper
- 5-day training, 1-day testing, 5-day buffer
- Z-score based entry signals
- Method-specific performance tracking
- Student's t-test for method comparison
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
import scipy.stats as stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from hurst import compute_Hc

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------
# Configuration and Data Structures
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
    # Experimental design
    training_days: int = 5
    testing_days: int = 1
    buffer_days: int = 5
    max_pairs_per_method: int = 5
    initial_capital_per_pair: float = 1000.0
    initial_capital_per_asset: float = 500.0
    
    # Trading parameters
    trading_fee: float = 0.0004  # 0.04% taker fee
    stop_loss_threshold: float = 0.05   # 5%
    take_profit_threshold: float = 0.02  # 2%
    
    # Statistical thresholds
    cointegration_p_threshold: float = 0.05
    correlation_threshold: float = 0.7
    distance_threshold: float = 0.1
    hurst_threshold: float = 0.5
    
    # Signal parameters
    z_score_entry_threshold: float = 1.0
    
    # Data parameters
    frequency: str = "1d"  # 1d, 15m, or 1h

@dataclass
class StatisticalResult:
    """Results from statistical analysis."""
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
    method: str  # Statistical method that selected this pair
    
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
    exit_reason: str = ""

@dataclass
class WindowResult:
    """Results for a single trading window."""
    window_start: pd.Timestamp
    training_end: pd.Timestamp
    testing_start: pd.Timestamp
    testing_end: pd.Timestamp
    
    # Statistical analysis results
    statistical_results: List[StatisticalResult] = field(default_factory=list)
    selected_pairs_by_method: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    
    # Trading results
    positions: List[Position] = field(default_factory=list)
    method_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

# ----------------------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------------------

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
)
log = logging.getLogger("paper_trading")

# ----------------------------------------------------------------------
# Statistical Analysis Methods (Paper Implementation)
# ----------------------------------------------------------------------

class PaperStatisticalAnalyzer:
    """Implements the six statistical methods from the paper."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def analyze_pair(self, data: pd.DataFrame, asset1: str, asset2: str) -> List[StatisticalResult]:
        """Run all six statistical methods on a pair."""
        results = []
        
        # Extract clean data
        pair_data = data[[asset1, asset2]].dropna()
        if len(pair_data) < 20:
            return results
        
        series1, series2 = pair_data[asset1], pair_data[asset2]
        
        # 1. Engle-Granger Cointegration
        results.append(self._engle_granger_cointegration(series1, series2, asset1, asset2))
        
        # 2. Pearson Correlation
        results.append(self._pearson_correlation(series1, series2, asset1, asset2))
        
        # 3. Euclidean Distance
        results.append(self._euclidean_distance(series1, series2, asset1, asset2))
        
        # 4. Fluctuation Behaviour
        results.append(self._fluctuation_behaviour(series1, series2, asset1, asset2))
        
        # 5. Hurst Exponent
        results.append(self._hurst_exponent(series1, series2, asset1, asset2))
        
        # 6. Stochastic Differential Residual
        results.append(self._stochastic_differential_residual(series1, series2, asset1, asset2))
        
        return results
    
    def _engle_granger_cointegration(self, s1: pd.Series, s2: pd.Series, 
                                   asset1: str, asset2: str) -> StatisticalResult:
        """Engle-Granger cointegration test."""
        try:
            # Perform cointegration test
            _, p_value, _ = coint(s1, s2)
            score = 1 - p_value if not np.isnan(p_value) else 0
            
            return StatisticalResult(
                pair=(asset1, asset2),
                method="cointegration",
                score=score,
                p_value=p_value,
                is_significant=p_value < self.config.cointegration_p_threshold
            )
        except:
            return StatisticalResult(
                pair=(asset1, asset2),
                method="cointegration",
                score=0,
                p_value=1.0,
                is_significant=False
            )
    
    def _pearson_correlation(self, s1: pd.Series, s2: pd.Series, 
                           asset1: str, asset2: str) -> StatisticalResult:
        """Pearson correlation coefficient."""
        try:
            corr = s1.corr(s2)
            if np.isnan(corr):
                corr = 0
            return StatisticalResult(
                pair=(asset1, asset2),
                method="correlation",
                score=abs(corr),
                is_significant=abs(corr) > self.config.correlation_threshold
            )
        except:
            return StatisticalResult(
                pair=(asset1, asset2),
                method="correlation",
                score=0,
                is_significant=False
            )
    
    def _euclidean_distance(self, s1: pd.Series, s2: pd.Series, 
                          asset1: str, asset2: str) -> StatisticalResult:
        """Euclidean distance between normalized series."""
        try:
            # Normalize both series
            norm1 = (s1 - s1.min()) / (s1.max() - s1.min())
            norm2 = (s2 - s2.min()) / (s2.max() - s2.min())
            
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((norm1 - norm2) ** 2)) / len(s1)
            
            # Convert to similarity score
            score = np.exp(-distance)
            
            return StatisticalResult(
                pair=(asset1, asset2),
                method="distance",
                score=score,
                is_significant=distance < self.config.distance_threshold
            )
        except:
            return StatisticalResult(
                pair=(asset1, asset2),
                method="distance",
                score=0,
                is_significant=False
            )
    
    def _fluctuation_behaviour(self, s1: pd.Series, s2: pd.Series, 
                             asset1: str, asset2: str) -> StatisticalResult:
        """Fluctuation behavior analysis."""
        try:
            # Calculate spread and zero crossings
            spread = s1 - s2
            zero_crossings = len(np.where(np.diff(np.sign(spread - spread.mean())))[0])
            crossings_per_day = zero_crossings / (len(spread) / 252)  # Annualized
            
            # Score based on crossing frequency
            score = min(crossings_per_day / 100, 1.0)  # Normalize
            
            return StatisticalResult(
                pair=(asset1, asset2),
                method="fluctuation",
                score=score,
                is_significant=zero_crossings > 5
            )
            
        except:
            return StatisticalResult(
                pair=(asset1, asset2),
                method="fluctuation",
                score=0,
                is_significant=False
            )
    
    def _hurst_exponent(self, s1: pd.Series, s2: pd.Series, 
                      asset1: str, asset2: str) -> StatisticalResult:
        """Hurst exponent for mean reversion detection."""
        try:
            # Use price ratio
            ratio = s1 / s2
            H, _, _ = compute_Hc(ratio.dropna(), kind='price', simplified=True)
            
            # Mean reversion score (lower H = better)
            score = max(0, (0.5 - H) * 2) if H < 0.5 else 0
            
            return StatisticalResult(
                pair=(asset1, asset2),
                method="hurst",
                score=score,
                is_significant=H < self.config.hurst_threshold
            )
        except:
            return StatisticalResult(
                pair=(asset1, asset2),
                method="hurst",
                score=0,
                is_significant=False
            )
    
    def _stochastic_differential_residual(self, s1: pd.Series, s2: pd.Series, 
                                        asset1: str, asset2: str) -> StatisticalResult:
        """Stochastic Differential Residual method."""
        try:
            # Calculate returns
            ret1 = s1.pct_change().dropna()
            ret2 = s2.pct_change().dropna()
            
            # Market proxy
            market = (ret1 + ret2) / 2
            
            # Calculate betas
            beta1 = np.cov(ret1, market)[0, 1] / np.var(market)
            beta2 = np.cov(ret2, market)[0, 1] / np.var(market)
            
            # Residual spread
            residual = ret1 - ret2 - (beta1 - beta2) * market
            
            # Test stationarity
            _, p_value, _, _, _, _ = adfuller(residual.dropna())
            score = 1 - p_value if not np.isnan(p_value) else 0
            
            return StatisticalResult(
                pair=(asset1, asset2),
                method="sdr",
                score=score,
                p_value=p_value,
                is_significant=p_value < 0.05
            )
        except:
            return StatisticalResult(
                pair=(asset1, asset2),
                method="sdr",
                score=0,
                p_value=1.0,
                is_significant=False
            )

# ----------------------------------------------------------------------
# Pair Selection
# ----------------------------------------------------------------------

class PairSelector:
    """Selects trading pairs based on statistical analysis results."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def select_pairs_by_method(self, results: List[StatisticalResult]) -> Dict[str, List[Tuple[str, str]]]:
        """Select top pairs for each statistical method."""
        method_groups = {}
        for result in results:
            if result.method not in method_groups:
                method_groups[result.method] = []
            method_groups[result.method].append(result)
        
        selected_pairs = {}
        
        # For each method, select top pairs
        for method, method_results in method_groups.items():
            method_results.sort(key=lambda x: x.score, reverse=True)
            top_pairs = [res.pair for res in method_results[:self.config.max_pairs_per_method]]
            selected_pairs[method] = top_pairs
        
        log.info(f"Selected pairs for {len(method_groups)} methods")
        return selected_pairs

# ----------------------------------------------------------------------
# Position Management
# ----------------------------------------------------------------------

class PositionManager:
    """Manages trading positions with risk controls."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def open_position(self, pair: Tuple[str, str], entry_date: pd.Timestamp, 
                     prices: Dict[str, float], signal_direction: int,
                     method: str) -> Position:
        """Open a new trading position."""
        asset1, asset2 = pair
        
        # Determine position type
        position_type = PositionType.LONG_SHORT if signal_direction > 0 else PositionType.SHORT_LONG
        
        # Calculate quantities
        quantity1 = self.config.initial_capital_per_asset / prices[asset1]
        quantity2 = self.config.initial_capital_per_asset / prices[asset2]
        
        # Entry fees
        entry_fees = (self.config.initial_capital_per_asset * 2) * self.config.trading_fee
        
        position = Position(
            pair=pair,
            position_type=position_type,
            entry_date=entry_date,
            entry_prices=prices.copy(),
            quantities={asset1: quantity1, asset2: quantity2},
            capital_allocated=self.config.initial_capital_per_pair,
            method=method,
            fees_paid=entry_fees
        )
        
        log.debug(f"Opened {position_type.value} position for {asset1}-{asset2} ({method})")
        return position
    
    def update_position(self, position: Position, current_date: pd.Timestamp, 
                       current_prices: Dict[str, float]) -> None:
        """Update position metrics and check for exit conditions."""
        if position.status != PositionStatus.OPEN:
            return
        
        asset1, asset2 = position.pair
        
        # Calculate current P&L
        if position.position_type == PositionType.LONG_SHORT:
            pnl1 = position.quantities[asset1] * (current_prices[asset1] - position.entry_prices[asset1])
            pnl2 = position.quantities[asset2] * (position.entry_prices[asset2] - current_prices[asset2])
        else:
            pnl1 = position.quantities[asset1] * (position.entry_prices[asset1] - current_prices[asset1])
            pnl2 = position.quantities[asset2] * (current_prices[asset2] - position.entry_prices[asset2])
        
        current_pnl = pnl1 + pnl2
        position.pnl = current_pnl
        
        # Update tracking metrics
        position.max_profit = max(position.max_profit, current_pnl)
        position.max_drawdown = min(position.max_drawdown, current_pnl)
        
        # Calculate holding period in days
        if self.config.frequency in ["1d", "15m"]:
            # For intraday, use hours held
            hours_held = (current_date - position.entry_date).total_seconds() / 3600
            position.days_held = hours_held / 24  # Convert to fractional days
        else:
            position.days_held = (current_date - position.entry_date).days
        
        # Check exit conditions
        pnl_percentage = current_pnl / position.capital_allocated
        
        # Stop loss
        if pnl_percentage <= -self.config.stop_loss_threshold:
            self.close_position(position, current_date, current_prices, "stop_loss")
        
        # Take profit
        elif pnl_percentage >= self.config.take_profit_threshold:
            self.close_position(position, current_date, current_prices, "take_profit")
        
        # Maximum holding period (testing + buffer)
        elif position.days_held >= (self.config.testing_days + self.config.buffer_days):
            self.close_position(position, current_date, current_prices, "max_holding")
    
    def close_position(self, position: Position, exit_date: pd.Timestamp, 
                      exit_prices: Dict[str, float], reason: str) -> None:
        """Close a trading position."""
        position.status = PositionStatus.CLOSED
        position.exit_date = exit_date
        position.exit_prices = exit_prices.copy()
        position.exit_reason = reason
        
        # Exit fees
        exit_fees = (self.config.initial_capital_per_asset * 2) * self.config.trading_fee
        position.fees_paid += exit_fees
        
        # Final P&L
        position.pnl -= position.fees_paid
        
        log.debug(f"Closed position {position.pair[0]}-{position.pair[1]} ({position.method}) "
                 f"on {exit_date} (reason: {reason}, P&L: ${position.pnl:.2f})")

# ----------------------------------------------------------------------
# Main Trading Pipeline
# ----------------------------------------------------------------------

class PaperTradingPipeline:
    """Main trading pipeline implementing the paper's experimental design."""
    
    def __init__(self, config: TradingConfig, data_path: Path):
        self.config = config
        self.data_path = data_path
        self.analyzer = PaperStatisticalAnalyzer(config)
        self.selector = PairSelector(config)
        self.position_manager = PositionManager(config)
        
        # Pipeline state
        self.window_results: List[WindowResult] = []
        self.method_returns: Dict[str, List[float]] = {
            "cointegration": [],
            "correlation": [],
            "distance": [],
            "fluctuation": [],
            "hurst": [],
            "sdr": []
        }
    
    def load_data(self, cryptos: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load cryptocurrency price data from Parquet files."""
        log.info(f"Loading {self.config.frequency} data for {len(cryptos)} cryptos")
        
        # Define data directory
        freq_dir = "1d" if self.config.frequency == "1d" else \
                  "15m" if self.config.frequency == "15m" else "15m"
        
        proc_dir = self.data_path / "processed" / freq_dir
        
        if not proc_dir.exists():
            log.error(f"Data directory not found: {proc_dir}")
            exit(1)
        
        dfs = []
        for crypto in cryptos:
            parquet_file = proc_dir / f"{crypto.lower()}.parquet"
            if parquet_file.exists():
                try:
                    df = pd.read_parquet(parquet_file)
                    if "close" in df.columns:
                        crypto_data = df[["close"]].rename(columns={"close": crypto})
                        dfs.append(crypto_data)
                except Exception as e:
                    log.warning(f"Error loading {crypto}: {e}")
        
        if not dfs:
            log.error("No data loaded")
            exit(1)
        
        # Merge data
        merged = pd.concat(dfs, axis=1)
        merged = merged.loc[start_date:end_date]
        
        log.info(f"Loaded data with shape: {merged.shape}")
        return merged
    
    def generate_trading_windows(self, start_date: str, end_date: str) -> List[WindowResult]:
        """Generate sliding windows per paper's experimental design."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        windows = []
        current_date = start
        
        while current_date <= end:
            training_end = current_date + pd.Timedelta(days=self.config.training_days)
            testing_start = training_end + pd.Timedelta(days=1)
            testing_end = testing_start + pd.Timedelta(days=self.config.testing_days + self.config.buffer_days)
            
            if testing_end > end:
                break
            
            window = WindowResult(
                window_start=current_date,
                training_end=training_end,
                testing_start=testing_start,
                testing_end=testing_end
            )
            windows.append(window)
            
            # Move to next day
            current_date += pd.Timedelta(days=1)
        
        log.info(f"Generated {len(windows)} trading windows")
        return windows
    
    def run_training_phase(self, window: WindowResult, data: pd.DataFrame, 
                          cryptos: List[str]) -> None:
        """Run the 5-day training phase for statistical analysis."""
        log.info(f"Training: {window.window_start} to {window.training_end}")
        
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
        
        log.info("Training completed")
    
    def generate_trading_signal(self, pair_data: pd.DataFrame, asset1: str, asset2: str) -> int:
        """Generate trading signal based on price ratio z-score."""
        if len(pair_data) < 10:
            return 0
        
        # Calculate price ratio
        ratio = pair_data[asset1] / pair_data[asset2]
        
        # Z-score calculation
        mean_ratio = ratio.mean()
        std_ratio = ratio.std()
        recent_ratio = ratio.iloc[-1]
        
        z_score = (recent_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
        
        # Generate signal
        if z_score > self.config.z_score_entry_threshold:
            return -1  # SHORT_LONG
        elif z_score < -self.config.z_score_entry_threshold:
            return 1   # LONG_SHORT
        else:
            return 0   # No signal
    
    def run_trading_phase(self, window: WindowResult, data: pd.DataFrame) -> None:
        """Run the trading phase with position management."""
        log.info(f"Trading: {window.testing_start} to {window.testing_end}")
        
        # Open positions for selected pairs
        entry_date = window.training_end  # Trade at end of training period
        if entry_date in data.index:
            current_prices = data.loc[entry_date].to_dict()
            
            for method, pairs in window.selected_pairs_by_method.items():
                for pair in pairs:
                    asset1, asset2 = pair
                    
                    # Generate trading signal
                    training_data = data.loc[window.window_start:window.training_end]
                    signal = self.generate_trading_signal(training_data, asset1, asset2)
                    
                    if signal != 0:
                        try:
                            position = self.position_manager.open_position(
                                pair, entry_date, current_prices, signal, method
                            )
                            window.positions.append(position)
                        except Exception as e:
                            log.warning(f"Failed to open position: {e}")
        
        # Update positions during testing and buffer period
        trading_data = data.loc[window.testing_start:window.testing_end]
        
        for timestamp, row in trading_data.iterrows():
            current_prices = row.to_dict()
            for position in window.positions:
                if position.status == PositionStatus.OPEN:
                    self.position_manager.update_position(position, timestamp, current_prices)
        
        # Close any remaining positions at end of buffer
        for position in window.positions:
            if position.status == PositionStatus.OPEN:
                self.position_manager.close_position(
                    position, 
                    window.testing_end, 
                    data.loc[window.testing_end].to_dict(), 
                    "buffer_end"
                )
        
        # Calculate method performance
        window.method_performance = self.calculate_method_performance(window.positions)
        
        # Track returns for t-tests
        for method, perf in window.method_performance.items():
            if "return" in perf:
                self.method_returns[method].append(perf["return"])
    
    def calculate_method_performance(self, positions: List[Position]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics per method."""
        method_perf = {}
        method_positions = {}
        
        # Group positions by method
        for position in positions:
            if position.method not in method_positions:
                method_positions[position.method] = []
            method_positions[position.method].append(position)
        
        # Calculate metrics per method
        for method, positions in method_positions.items():
            total_pnl = sum(p.pnl for p in positions)
            total_fees = sum(p.fees_paid for p in positions)
            net_pnl = total_pnl - total_fees
            capital_allocated = len(positions) * self.config.initial_capital_per_pair
            return_pct = net_pnl / capital_allocated if capital_allocated > 0 else 0
            
            profitable = [p for p in positions if p.pnl > 0]
            success_rate = len(profitable) / len(positions) if positions else 0
            
            method_perf[method] = {
                "positions": len(positions),
                "total_pnl": total_pnl,
                "total_fees": total_fees,
                "net_pnl": net_pnl,
                "return": return_pct,
                "success_rate": success_rate
            }
        
        return method_perf
    
    def run_backtest(self, cryptos: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """Run the complete backtest."""
        log.info(f"Starting backtest: {start_date} to {end_date}")
        start_time = time.time()
        
        # Load data
        data = self.load_data(cryptos, start_date, end_date)
        
        # Generate trading windows
        windows = self.generate_trading_windows(start_date, end_date)
        
        # Process each window
        for i, window in enumerate(windows):
            log.info(f"Processing window {i+1}/{len(windows)}")
            
            try:
                self.run_training_phase(window, data, cryptos)
                self.run_trading_phase(window, data)
                self.window_results.append(window)
            except Exception as e:
                log.error(f"Window failed: {e}")
        
        # Generate final results
        results = self.generate_results_summary()
        
        # Perform t-tests
        results["t_tests"] = self.perform_t_tests()
        
        log.info(f"Backtest completed in {time.time()-start_time:.2f}s")
        return results
    
    def generate_results_summary(self) -> Dict[str, Any]:
        """Generate comprehensive results summary."""
        summary = {
            "total_windows": len(self.window_results),
            "total_positions": sum(len(w.positions) for w in self.window_results),
            "method_performance": {},
            "overall_performance": {
                "total_pnl": 0,
                "total_fees": 0,
                "net_pnl": 0,
                "total_return": 0
            }
        }
        
        # Aggregate method performance
        method_metrics = {}
        for method in self.method_returns.keys():
            returns = self.method_returns[method]
            if returns:
                avg_return = np.mean(returns)
                win_rate = sum(1 for r in returns if r > 0) / len(returns)
            else:
                avg_return = 0
                win_rate = 0
            
            method_metrics[method] = {
                "avg_return": avg_return,
                "win_rate": win_rate,
                "total_return": sum(returns),
                "num_windows": len(returns)
            }
            
            # Add to overall
            summary["overall_performance"]["total_pnl"] += method_metrics[method]["total_return"]
        
        summary["method_performance"] = method_metrics
        
        # Calculate overall return
        total_capital = len(self.window_results) * self.config.max_pairs_per_method * 6 * self.config.initial_capital_per_pair
        if total_capital > 0:
            summary["overall_performance"]["total_return"] = \
                summary["overall_performance"]["total_pnl"] / total_capital
        
        return summary
    
    def perform_t_tests(self) -> Dict[str, Dict[str, float]]:
        """Perform Student's t-tests between method returns."""
        t_test_results = {}
        methods = list(self.method_returns.keys())
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                returns1 = self.method_returns[method1]
                returns2 = self.method_returns[method2]
                
                # Ensure we have comparable data
                if len(returns1) < 2 or len(returns2) < 2:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(returns1, returns2, equal_var=False)
                
                # Determine significance
                significant = {
                    "1%": p_value < 0.01,
                    "5%": p_value < 0.05,
                    "10%": p_value < 0.10
                }
                
                key = f"{method1}_vs_{method2}"
                t_test_results[key] = {
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "significant": significant
                }
        
        return t_test_results

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save results to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Config-based filename
        freq = self.config.frequency.replace("m", "min")
        filename = f"backtest_results_{freq}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        log.info(f"Results saved to {output_path}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of results."""
        print("\n" + "=" * 80)
        print("CRYPTOCURRENCY PAIRS TRADING BACKTEST RESULTS")
        print(f"Frequency: {self.config.frequency} | Period: {results['start_date']} to {results['end_date']}")
        print("=" * 80)
        
        # Overall performance
        overall = results["summary"]["overall_performance"]
        print(f"\nOverall Performance:")
        print(f"  Windows: {results['summary']['total_windows']}")
        print(f"  Positions: {results['summary']['total_positions']}")
        print(f"  Total Return: {overall['total_return']:.2%}")
        print(f"  Net P&L: ${overall['net_pnl']:,.2f}")
        
        # Method performance
        print("\nMethod Performance:")
        print("{:<15} {:<12} {:<12} {:<10}".format(
            "Method", "Avg Return", "Win Rate", "Total Return"
        ))
        for method, perf in results["summary"]["method_performance"].items():
            print("{:<15} {:<12.4%} {:<12.2%} {:<10.2%}".format(
                method, perf["avg_return"], perf["win_rate"], perf["total_return"]
            ))
        
        # T-test results
        print("\nT-Test Results (Method Comparison):")
        for comparison, test in results["t_tests"].items():
            stars = ""
            if test["p_value"] < 0.01: stars = "***"
            elif test["p_value"] < 0.05: stars = "**"
            elif test["p_value"] < 0.10: stars = "*"
            
            print(f"{comparison}: t-stat={test['t_stat']:.4f}, p={test['p_value']:.4f} {stars}")
        
        print("=" * 80)


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

def main():
    """Main execution function."""
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
    
    # Backtest period (Jan 1, 2022 to Mar 31, 2022)
    start_date = "2022-01-01"
    end_date = "2022-03-31"
    
    # Data path
    data_path = Path("./data")
    output_dir = Path("./results")
    
    # Run for each frequency
    for freq in ["1d", "115m", "1h"]:
        # Configuration
        config = TradingConfig(
            frequency=freq,
            training_days=5,
            testing_days=1,
            buffer_days=5,
            max_pairs_per_method=5
        )
        
        # Initialize pipeline
        pipeline = PaperTradingPipeline(config, data_path)
        
        try:
            # Run backtest
            results = pipeline.run_backtest(cryptos, start_date, end_date)
            
            # Add metadata
            results["start_date"] = start_date
            results["end_date"] = end_date
            results["cryptos"] = cryptos
            results["config"] = config.__dict__
            
            # Save and print results
            pipeline.save_results(results, output_dir)
            pipeline.print_summary(results)
            
        except Exception as e:
            log.error(f"Backtest failed for {freq}: {e}")


if __name__ == "__main__":
    # Set up results directory
    Path("./results").mkdir(exist_ok=True)
    
    # Run the main function
    main()
    