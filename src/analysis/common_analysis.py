#!/usr/bin/env python3
"""
Common utilities for crypto pair analysis (cointegration, correlation, distance, etc.)
Shared components that can be imported by specific analysis scripts.
"""

import json
import warnings
import sys
import logging
from pathlib import Path
from itertools import combinations
import time
from typing import List, Tuple, NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.metrics import pairwise_distances

# ----------------------------------------------------------------------
# LOGGING SETUP
# ----------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

def setup_logging(logger_name="analysis", verbose=False):
    """Setup standardized logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FMT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Suppress numpy warnings for cleaner output
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    return logging.getLogger(logger_name)

# ----------------------------------------------------------------------
# CONFIGURATION MANAGEMENT
# ----------------------------------------------------------------------
def create_default_config():
    """Create default configuration dictionary."""
    return {
        "cryptos": [
            "ETHUSDT", "BNBUSDT", "BTCUSDT", "MATICUSDT", "SHIBUSDT", "SANDUSDT", 
            "SOLUSDT", "GALAUSDT", "XRPUSDT", "AVAXUSDT", "DOTUSDT", "ADAUSDT", 
            "DOGEUSDT", "MANAUSDT", "FTMUSDT", "NEARUSDT", "TRXUSDT", "FILUSDT", 
            "LINKUSDT", "MBOXUSDT", "LTCUSDT", "ATOMUSDT", "CTXCUSDT", "CRVUSDT", 
            "EGLDUSDT", "EOSUSDT", "SUSHIUSDT", "ALICEUSDT", "AXSUSDT", "ICPUSDT"
        ],
        "intervals": ["1m", "5m", "1h"],
        "start_date": "2022-01-01",
        "end_date": "2022-03-31",
        "window_days": 30,
        "step_days": 5,
        "min_rows_per_window": 50,
        "p_threshold": 0.05
    }

def find_config_file(config_path=None, script_file=None):
    """
    Find configuration file with fallback logic.
    
    Args:
        config_path: Explicitly provided config path
        script_file: Path to the calling script (for relative path calculation)
    
    Returns:
        Path to config file
    """
    if script_file:
        script_path = Path(script_file).resolve()
        repo_root = script_path.parents[2] if len(script_path.parents) >= 2 else Path.cwd()
    else:
        repo_root = Path.cwd()
    
    # Try provided path first
    if config_path and Path(config_path).exists():
        return Path(config_path), repo_root
    
    # Try default locations
    default_path = repo_root / "config" / "config.json"
    fallback_path = Path.cwd() / "config" / "config.json"
    
    if default_path.exists():
        return default_path, repo_root
    elif fallback_path.exists():
        return fallback_path, repo_root
    else:
        # Create default config
        cfg_path = Path.cwd() / "config.json"
        with open(cfg_path, 'w') as f:
            json.dump(create_default_config(), f, indent=2)
        return cfg_path, repo_root

def load_config(config_path=None, script_file=None):
    """Load and validate configuration with fallback logic."""
    cfg_path, repo_root = find_config_file(config_path, script_file)
    
    try:
        with open(cfg_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    
    # Validate required fields
    required_fields = ["cryptos", "intervals", "start_date", "end_date"]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    
    # Set defaults
    defaults = {
        "window_days": 30,
        "step_days": 5,
        "min_rows_per_window": 50,
        "p_threshold": 0.05
    }
    for key, default_value in defaults.items():
        config.setdefault(key, default_value)
    
    return config, repo_root

# ----------------------------------------------------------------------
# DATA PREPROCESSING
# ----------------------------------------------------------------------
def preprocess_series(s1, s2, method='log_returns'):
    """
    Preprocess price series to avoid numerical issues.
    
    Args:
        s1, s2: Price series
        method: 'log_returns', 'log_prices', 'standardize', 'diff', or 'none'
    
    Returns:
        Tuple of processed series
    """
    if method == 'log_returns':
        # Use log returns (stationary by nature)
        s1_proc = np.log(s1).diff().dropna()
        s2_proc = np.log(s2).diff().dropna()
        # Align series
        common_idx = s1_proc.index.intersection(s2_proc.index)
        return s1_proc.loc[common_idx], s2_proc.loc[common_idx]
    
    elif method == 'log_prices':
        # Use log prices (reduces scaling issues)
        s1_proc = np.log(s1)
        s2_proc = np.log(s2)
        return s1_proc, s2_proc
    
    elif method == 'standardize':
        # Standardize to mean=0, std=1
        scaler1, scaler2 = StandardScaler(), StandardScaler()
        s1_proc = pd.Series(
            scaler1.fit_transform(s1.values.reshape(-1, 1)).flatten(),
            index=s1.index
        )
        s2_proc = pd.Series(
            scaler2.fit_transform(s2.values.reshape(-1, 1)).flatten(),
            index=s2.index
        )
        return s1_proc, s2_proc
    
    elif method == 'diff':
        # Simple differencing
        s1_proc = s1.diff().dropna()
        s2_proc = s2.diff().dropna()
        common_idx = s1_proc.index.intersection(s2_proc.index)
        return s1_proc.loc[common_idx], s2_proc.loc[common_idx]
    
    else:  # method == 'none'
        return s1, s2

def check_data_quality(s1, s2, pair_tag=""):
    """Check for data quality issues that could cause numerical problems."""
    issues = []
    
    # Check for infinite values
    if np.isinf(s1).any() or np.isinf(s2).any():
        issues.append("infinite values")
    
    # Check for extreme values (more than 10 orders of magnitude difference)
    if s1.std() > 0 and s2.std() > 0:
        ratio = max(s1.std(), s2.std()) / min(s1.std(), s2.std())
        if ratio > 1e10:
            issues.append(f"extreme scaling (ratio: {ratio:.2e})")
    
    # Check for perfect correlation (multicollinearity)
    if len(s1) > 1 and len(s2) > 1:
        corr = np.corrcoef(s1, s2)[0, 1]
        if abs(corr) > 0.9999:
            issues.append(f"near-perfect correlation ({corr:.6f})")
    
    # Check for constant series
    if s1.std() < 1e-10 or s2.std() < 1e-10:
        issues.append("constant/near-constant series")
    
    if issues:
        return False, issues
    
    return True, []

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
def find_data_file(base_path, symbol, interval):
    """
    Find data file for given symbol and interval in processed2 Parquet directory.
    Returns:
        Path to file or None if not found
    """
    data_path = base_path / "data" / "processed2" / interval
    sym_lower = symbol.lower()

    possible_files = [
        data_path / f"{sym_lower}_{interval}_2022_Q1.parquet",
        data_path / f"{sym_lower}_{interval}.parquet",
        data_path / f"{symbol}_{interval}_2022_Q1.parquet",
        data_path / f"{symbol.upper()}_{interval}_2022_Q1.parquet",
        data_path / f"{symbol}_{interval}.parquet",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path
    return None

def load_crypto_data(file_path, symbol):
    """
    Load cryptocurrency data from CSV or Parquet file.
    
    Returns:
        DataFrame with datetime index and close price column
    """
    # Load file
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_parquet(file_path)
    
    # Handle timestamp column
    timestamp_cols = ['timestamp', 'time', 'datetime', 'date', 'open_time', 'kline_open_time']
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col:
        # Convert timestamp to datetime index
        if df[timestamp_col].dtype in ['int64', 'float64']:
            # Assume milliseconds timestamp
            df['datetime'] = pd.to_datetime(df[timestamp_col], unit='ms')
        else:
            df['datetime'] = pd.to_datetime(df[timestamp_col])
        df.set_index('datetime', inplace=True)
    else:
        # Try to use existing index if it looks like datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Cannot create datetime index for {file_path}")
    
    # Find close price column
    close_cols = ['close', 'Close', 'CLOSE', 'close_price', 'price']
    close_col = None
    for col in close_cols:
        if col in df.columns:
            close_col = col
            break
    
    if not close_col:
        raise ValueError(f"No close price column found in {file_path}. Available: {list(df.columns)}")
    
    # Extract and clean close data
    close_data = df[[close_col]].rename(columns={close_col: symbol})
    close_data[symbol] = pd.to_numeric(close_data[symbol], errors='coerce')
    
    if close_data.empty or close_data.isna().all().iloc[0]:
        raise ValueError(f"Empty or all-NaN data in {file_path}")
    
    return close_data

def load_interval_data(base_path, cryptos, interval, log=None):
    """Load data for all cryptos in a specific interval."""
    if log is None:
        log = logging.getLogger(__name__)
    
    dfs, missing = [], []
    
    for symbol in cryptos:
        file_path = find_data_file(base_path, symbol, interval)
        
        if file_path:
            try:
                close_data = load_crypto_data(file_path, symbol)
                dfs.append(close_data)
                log.debug(f"Loaded {symbol} from {file_path.name}: {len(close_data)} rows")
            except Exception as e:
                log.error(f"Error loading {file_path}: {e}")
                missing.append(symbol)
        else:
            log.debug(f"No file found for {symbol} in interval {interval}")
            missing.append(symbol)

    if missing:
        log.warning(f"[{interval}] Missing {len(missing)}/{len(cryptos)} symbols: {', '.join(missing)}")

    if not dfs:
        raise FileNotFoundError(f"No valid data files found for interval {interval}")

    # Merge and clean
    merged = pd.concat(dfs, axis=1).sort_index()
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(how='all')
    
    if merged.empty:
        raise ValueError(f"No valid data after cleaning for interval {interval}")
    
    log.info(f"[{interval}] Loaded {merged.shape[1]} symbols × {merged.shape[0]} rows")
    log.info(f"[{interval}] Date range: {merged.index.min()} to {merged.index.max()}")
    return merged

# ----------------------------------------------------------------------
# WINDOW GENERATION (LEGACY)
# ----------------------------------------------------------------------
def generate_windows(start_date, end_date, window_days, step_days):
    """Generate sliding windows with validation."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    if start >= end:
        raise ValueError(f"Start date {start_date} must be before end date {end_date}")
    
    windows = []
    current = start
    
    while current + pd.Timedelta(days=window_days - 1) <= end:
        window_end = current + pd.Timedelta(days=window_days - 1)
        windows.append((current, window_end))
        current += pd.Timedelta(days=step_days)
    
    return windows

# ----------------------------------------------------------------------
# ENHANCED ROLLING-WINDOW FRAMEWORK
# ----------------------------------------------------------------------
class WindowPhase(NamedTuple):
    """Structure for a complete walk-forward window cycle."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    trade_start: pd.Timestamp
    trade_end: pd.Timestamp
    buffer_start: pd.Timestamp
    buffer_end: pd.Timestamp
    iteration: int

def generate_walkforward_windows_strict(
    start_date: str, 
    end_date: str, 
    training_days: int = 5,
    trading_days: int = 1,
    buffer_days: int = 5,
    step_days: int = 1,
    log: logging.Logger = None
) -> List[WindowPhase]:
    """
    Generate walk-forward windows following the exact specification:
    - Training: 5 trading days (past data only)
    - Trading: 1 day immediately after training
    - Buffer: ≤5 days for position expiry
    - Step: 1 day forward between iterations
    
    Args:
        start_date: Overall analysis start date
        end_date: Overall analysis end date  
        training_days: Training window length (default: 5)
        trading_days: Trading window length (default: 1)
        buffer_days: Buffer window length (default: 5)
        step_days: Days to advance between iterations (default: 1)
        log: Logger instance
    
    Returns:
        List of WindowPhase objects with all window boundaries
    """
    if log is None:
        log = logging.getLogger(__name__)
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    if start >= end:
        raise ValueError(f"Start date {start_date} must be before end date {end_date}")
    
    windows = []
    current = start + pd.Timedelta(days=training_days)  # Start with enough data for training
    iteration = 0
    
    while current <= end:
        # Calculate all window boundaries
        train_start = current - pd.Timedelta(days=training_days)
        train_end = current - pd.Timedelta(days=1)
        
        trade_start = current
        trade_end = trade_start + pd.Timedelta(days=trading_days - 1)
        
        buffer_start = trade_end + pd.Timedelta(days=1)
        buffer_end = buffer_start + pd.Timedelta(days=buffer_days - 1)
        
        # Check if we have enough data left
        if buffer_end > end:
            log.info(f"Stopping at iteration {iteration}: buffer window extends beyond end date")
            break
            
        # Ensure we have enough historical data for training
        if train_start < start:
            current += pd.Timedelta(days=step_days)
            continue
            
        windows.append(WindowPhase(
            train_start=train_start,
            train_end=train_end,
            trade_start=trade_start,
            trade_end=trade_end,
            buffer_start=buffer_start,
            buffer_end=buffer_end,
            iteration=iteration
        ))
        
        current += pd.Timedelta(days=step_days)
        iteration += 1
    
    total_days = (end - start).days
    log.info(f"Generated {len(windows)} walk-forward windows over {total_days} days")
    log.info(f"Window structure: {training_days}d train + {trading_days}d trade + {buffer_days}d buffer")
    
    if windows:
        log.info(f"First window: {windows[0].train_start.date()} to {windows[0].buffer_end.date()}")
        log.info(f"Last window: {windows[-1].train_start.date()} to {windows[-1].buffer_end.date()}")
    
    return windows

def validate_window_data_requirements(
    window: WindowPhase, 
    data: pd.DataFrame, 
    interval: str,
    min_bars_per_day: dict = None
) -> Tuple[bool, str]:
    """
    Validate that sufficient data exists for each phase of the window.
    
    Args:
        window: WindowPhase object
        data: Full dataset with datetime index
        interval: Data interval ('1m', '5m', '1h')
        min_bars_per_day: Expected bars per day for each interval
    
    Returns:
        (is_valid, message)
    """
    if min_bars_per_day is None:
        min_bars_per_day = {
            '1m': 1440,   # 24 * 60 minutes
            '5m': 288,    # 24 * 12 five-minute periods  
            '1h': 24,     # 24 hours
            '60m': 24     # Alternative hour notation
        }
    
    expected_per_day = min_bars_per_day.get(interval, 24)
    
    # Check training window data
    train_data = data.loc[window.train_start:window.train_end]
    expected_train_bars = expected_per_day * 5  # 5 training days
    
    if len(train_data) < expected_train_bars * 0.8:  # Allow 20% tolerance
        return False, f"Insufficient training data: {len(train_data)} < {expected_train_bars * 0.8:.0f}"
    
    # Check trading window data  
    trade_data = data.loc[window.trade_start:window.trade_end]
    expected_trade_bars = expected_per_day * 1  # 1 trading day
    
    if len(trade_data) < expected_trade_bars * 0.5:  # Allow 50% tolerance for single day
        return False, f"Insufficient trading data: {len(trade_data)} < {expected_trade_bars * 0.5:.0f}"
    
    return True, f"Valid: {len(train_data)} train + {len(trade_data)} trade bars"

def extract_window_phase_data(
    data: pd.DataFrame, 
    window: WindowPhase, 
    phase: str = 'train'
) -> pd.DataFrame:
    """
    Extract data for a specific phase of the window.
    
    Args:
        data: Full dataset
        window: WindowPhase object
        phase: 'train', 'trade', or 'buffer'
    
    Returns:
        DataFrame slice for the specified phase
    """
    if phase == 'train':
        return data.loc[window.train_start:window.train_end].copy()
    elif phase == 'trade':
        return data.loc[window.trade_start:window.trade_end].copy()
    elif phase == 'buffer':
        return data.loc[window.buffer_start:window.buffer_end].copy()
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 'train', 'trade', or 'buffer'")

def calculate_window_statistics(windows: List[WindowPhase]) -> dict:
    """Calculate statistics about the window structure."""
    if not windows:
        return {}
    
    total_period = (windows[-1].buffer_end - windows[0].train_start).days
    total_iterations = len(windows)
    
    # Calculate coverage
    training_days = sum((w.train_end - w.train_start).days + 1 for w in windows)
    trading_days = sum((w.trade_end - w.trade_start).days + 1 for w in windows) 
    buffer_days = sum((w.buffer_end - w.buffer_start).days + 1 for w in windows)
    
    return {
        'total_iterations': total_iterations,
        'total_period_days': total_period,
        'total_training_days': training_days,
        'total_trading_days': trading_days, 
        'total_buffer_days': buffer_days,
        'avg_days_between_windows': total_period / total_iterations if total_iterations > 1 else 0,
        'first_window_start': windows[0].train_start,
        'last_window_end': windows[-1].buffer_end
    }

# ----------------------------------------------------------------------
# WALK-FORWARD WINDOW GENERATION (LEGACY)
# ----------------------------------------------------------------------
def generate_walkforward_windows(start_date, end_date, train_days, trade_days, buffer_days, step_days=1):
    """
    Generate training/trading/buffer windows for walk-forward analysis.
    
    Args:
        start_date, end_date: Overall date range
        train_days: Training window length (days)
        trade_days: Trading window length (days)
        buffer_days: Buffer window length (days)
        step_days: Days to advance between iterations
    
    Returns:
        List of tuples: (train_start, train_end, trade_start, trade_end, buffer_end)
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    windows = []
    
    current = start
    while current < end:
        train_end = current + pd.Timedelta(days=train_days - 1)
        trade_start = train_end + pd.Timedelta(days=1)
        trade_end = trade_start + pd.Timedelta(days=trade_days - 1)
        buffer_end = trade_end + pd.Timedelta(days=buffer_days)
        
        if buffer_end > end:
            break
            
        windows.append((
            current, 
            train_end,
            trade_start,
            trade_end,
            buffer_end
        ))
        current += pd.Timedelta(days=step_days)
    
    return windows

# ----------------------------------------------------------------------
# PAIR PROCESSING
# ----------------------------------------------------------------------
def get_valid_pairs(data, cryptos):
    """Get all valid cryptocurrency pairs from the loaded data."""
    available_symbols = []
    for symbol in cryptos:
        # Case-insensitive column matching
        matching_col = next((col for col in data.columns if col.lower() == symbol.lower()), None)
        if matching_col:
            available_symbols.append((symbol, matching_col))
    
    return list(combinations(available_symbols, 2))

def extract_pair_data(data, symbol_info1, symbol_info2, window_start, window_end, min_rows):
    """
    Extract and validate pair data for a specific window.
    
    Args:
        data: Full dataset
        symbol_info1, symbol_info2: Tuples of (original_name, column_name)
        window_start, window_end: Window boundaries
        min_rows: Minimum required rows
    
    Returns:
        (pair_data, metadata) or (None, None) if invalid
    """
    symbol1, col1 = symbol_info1
    symbol2, col2 = symbol_info2
    
    window_slice = data.loc[window_start:window_end]
    pair_df = window_slice[[col1, col2]].dropna()
    
    if len(pair_df) < min_rows:
        return None, None
    
    metadata = {
        "symbol1": symbol1,
        "symbol2": symbol2,
        "col1": col1,
        "col2": col2,
        "rows": len(pair_df),
        "window_start": window_start,
        "window_end": window_end,
        "tag": f"{symbol1}-{symbol2} {window_start.date()}→{window_end.date()}"
    }
    
    return pair_df, metadata

def extract_pair_data_from_window(data, symbol_info1, symbol_info2, window: WindowPhase, phase: str = 'train', min_rows: int = 50):
    """
    Extract and validate pair data for a specific window phase.
    
    Args:
        data: Full dataset
        symbol_info1, symbol_info2: Tuples of (original_name, column_name)
        window: WindowPhase object
        phase: 'train', 'trade', or 'buffer'
        min_rows: Minimum required rows
    
    Returns:
        (pair_data, metadata) or (None, None) if invalid
    """
    symbol1, col1 = symbol_info1
    symbol2, col2 = symbol_info2
    
    phase_data = extract_window_phase_data(data, window, phase)
    pair_df = phase_data[[col1, col2]].dropna()
    
    if len(pair_df) < min_rows:
        return None, None
    
    metadata = {
        "symbol1": symbol1,
        "symbol2": symbol2,
        "col1": col1,
        "col2": col2,
        "rows": len(pair_df),
        "window_iteration": window.iteration,
        "phase": phase,
        "window_start": getattr(window, f"{phase}_start"),
        "window_end": getattr(window, f"{phase}_end"),
        "tag": f"{symbol1}-{symbol2} {phase} iter{window.iteration}"
    }
    
    return pair_df, metadata

# ----------------------------------------------------------------------
# COMMON METRICS
# ----------------------------------------------------------------------
def calculate_common_metrics(s1, s2):
    """Calculate metrics common to all analysis methods."""
    correlation = s1.corr(s2) if len(s1) > 1 else 0.0
    vol1, vol2 = s1.std(), s2.std()
    volatility_ratio = vol1 / vol2 if vol2 > 0 else 0.0
    
    return {
        "correlation": round(correlation, 4),
        "volatility_ratio": round(volatility_ratio, 4),
        "volatility_1": round(vol1, 6),
        "volatility_2": round(vol2, 6),
    }

# ----------------------------------------------------------------------
# OUTPUT UTILITIES
# ----------------------------------------------------------------------
def save_results(results, output_dir, filename, log=None):
    """Save results to CSV with proper directory creation."""
    if log is None:
        log = logging.getLogger(__name__)
    
    if not results:
        log.warning("No results to save")
        return None
    
    df = pd.DataFrame(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    
    return output_path

def create_analysis_stats():
    """Create a statistics tracking dictionary."""
    return {
        "tested": 0,
        "too_short": 0,
        "failed": 0,
        "successful": 0,
        "start_time": time.time()
    }

def update_stats(stats, result_type):
    """Update statistics counter."""
    if result_type in stats:
        stats[result_type] += 1

def log_final_stats(stats, interval, log, additional_info=""):
    """Log final statistics for an interval."""
    processing_time = time.time() - stats["start_time"]
    log.info(
        f"[{interval}] Completed: {stats['tested']} pairs tested, "
        f"{stats['successful']} successful, {stats['too_short']} skipped, "
        f"{stats['failed']} failed in {processing_time:.2f}s {additional_info}"
    )

# ----------------------------------------------------------------------
# COMMAND LINE UTILITIES
# ----------------------------------------------------------------------
def add_common_arguments(parser):
    """Add common command line arguments."""
    parser.add_argument(
        "-c", "--config",
        help="Path to config.json file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List available data files and exit"
    )
    return parser

def list_data_files(repo_root, log):
    """List available data files."""
    data_path = repo_root / "data" / "raw2"
    if data_path.exists():
        log.info(f"Files in {data_path}:")
        for file_path in sorted(data_path.glob("*")):
            log.info(f"  {file_path.name}")
    else:
        log.error(f"Data directory not found: {data_path}")

# ----------------------------------------------------------------------
# GRANGER CAUSALITY & METRICS
# ----------------------------------------------------------------------
def btc_granger_test(alt_series, btc_series, maxlags=4):
    """
    Test if BTC Granger-causes altcoin with sign check on beta coefficients.
    
    Returns:
        dict: Test results with p-value and coefficient sign info
    """
    try:
        # Align and drop missing values
        df = pd.DataFrame({'alt': alt_series, 'btc': btc_series}).dropna()
        if len(df) < maxlags + 5:  # Minimum data requirement
            return None
        
        # Fit VAR model
        model = VAR(df)
        results = model.fit(maxlags=maxlags, ic='aic')
        
        # Get BTC coefficients in alt equation
        alt_coeffs = results.params.get('alt', pd.Series())
        btc_lag_coeffs = alt_coeffs.filter(like='btc', axis=0)
        
        # Perform Granger causality test
        test_res = results.test_causality('alt', ['btc'], kind='wald')
        
        return {
            'gc_pvalue': test_res.pvalue,
            'gc_reject': test_res.pvalue < 0.05,
            'btc_coeffs': btc_lag_coeffs.to_dict(),
            'any_positive': any(btc_lag_coeffs > 0)
        }
        
    except Exception as e:
        return {'error': str(e)} 
    

def calculate_cointegration(s1, s2):
    """Calculate cointegration metrics between two series."""
    try:
        # Use log prices for cointegration test
        s1_log, s2_log = np.log(s1), np.log(s2)
        coint_res = coint(s1_log, s2_log, autolag='BIC')
        return {
            'coint_tstat': coint_res[0],
            'coint_pvalue': coint_res[1],
            'coint_crit_vals': coint_res[2]
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_distance(s1, s2):
    """Calculate normalized distance metric between two series."""
    try:
        # Normalize log prices
        s1_log, s2_log = np.log(s1), np.log(s2)
        s1_norm = (s1_log - s1_log.mean()) / s1_log.std()
        s2_norm = (s2_log - s2_log.mean()) / s2_log.std()
        return np.mean(np.abs(s1_norm - s2_norm))
    except:
        return np.nan

# ----------------------------------------------------------------------
# PAIR METRICS COMPUTATION
# ----------------------------------------------------------------------
def compute_pair_metrics(pair_df, btc_symbol='BTCUSDT'):
    """
    Compute full set of metrics for a crypto pair.
    
    Args:
        pair_df: DataFrame with two price series
        btc_symbol: Identifier for Bitcoin
    
    Returns:
        dict: Comprehensive metrics dictionary
    """
    sym1, sym2 = pair_df.columns
    s1, s2 = pair_df[sym1], pair_df[sym2]
    
    # Base metrics
    metrics = {
        'symbol1': sym1,
        'symbol2': sym2,
        'num_observations': len(pair_df),
        **calculate_common_metrics(s1, s2)
    }
    
    # Cointegration metrics
    metrics.update(calculate_cointegration(s1, s2))
    
    # Distance metric
    metrics['distance'] = calculate_distance(s1, s2)
    
    # Granger causality if BTC is in pair
    if btc_symbol in (sym1, sym2):
        alt_sym = sym2 if sym1 == btc_symbol else sym1
        btc_sym = sym1 if sym1 == btc_symbol else sym2
        
        # Use log returns for Granger test
        alt_ret = np.log(pair_df[alt_sym]).diff().dropna()
        btc_ret = np.log(pair_df[btc_sym]).diff().dropna()
        
        if not alt_ret.empty and not btc_ret.empty:
            gc_res = btc_granger_test(alt_ret, btc_ret)
            if gc_res:
                metrics.update({
                    'gc_pvalue': gc_res['gc_pvalue'],
                    'gc_reject': gc_res['gc_reject'],
                    'gc_any_positive': gc_res['any_positive'],
                    'gc_coeffs': gc_res['btc_coeffs']
                })
    
    return metrics

# ----------------------------------------------------------------------
# BACKTESTING UTILITIES
# ----------------------------------------------------------------------
def calculate_spread(price_a, price_b, hedge_ratio):
    """Calculate spread series for two price series."""
    return np.log(price_a) - hedge_ratio * np.log(price_b)

def calculate_zscore(spread, lookback=24*60):
    """Calculate rolling z-score of spread."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std