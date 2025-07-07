#!/usr/bin/env python3
"""
Find low-DTW-distance crypto pairs and dump one CSV per interval.

Standardized version that outputs comparable CSV format with cointegration analysis.
"""

import json
import argparse
import sys
import logging
import warnings
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from dtaidistance import dtw

# Suppress DTW warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="dtaidistance")

# ----------------------------------------------------------------------
# 0 ─── Data Classes & Configuration ───────────────────────────────────
# ----------------------------------------------------------------------
@dataclass
class DTWResult:
    """Container for DTW calculation results."""
    method: str
    interval: str
    window_start: str
    window_end: str
    asset1: str
    asset2: str
    rows: int
    dtw_distance: float
    dtw_normalized: float
    correlation: float
    volatility_ratio: float
    similarity_score: float

# ----------------------------------------------------------------------
# 1 ─── Logging Setup ──────────────────────────────────────────────────
# ----------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dtw")

# ----------------------------------------------------------------------
# 2 ─── Configuration Management ───────────────────────────────────────
# ----------------------------------------------------------------------
def load_config(config_path: Optional[Path] = None) -> Tuple[Dict[str, Any], Path]:
    """Load and validate configuration with fallback logic."""
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    
    # Try provided path first
    if config_path and Path(config_path).exists():
        cfg_path = Path(config_path)
    else:
        # Try default locations
        default_path = repo_root / "config" / "config.json"
        fallback_path = Path.cwd() / "config" / "config.json"
        
        if default_path.exists():
            cfg_path = default_path
        elif fallback_path.exists():
            cfg_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Config file not found. Searched: {config_path}, {default_path}, {fallback_path}"
            )
    
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
    config.setdefault("window_days", 30)
    config.setdefault("step_days", 5)
    config.setdefault("dtw_window_size", None)
    config.setdefault("min_rows_per_window", 50)
    config.setdefault("base_path", "data/processed")
    config.setdefault("dtw_threshold", 10.0)  # For similarity classification
    
    # Validate date format
    try:
        pd.Timestamp(config["start_date"])
        pd.Timestamp(config["end_date"])
    except ValueError as e:
        raise ValueError(f"Invalid date format in config: {e}")
    
    log.info(f"Loaded config from {cfg_path}")
    return config, repo_root

# ----------------------------------------------------------------------
# 3 ─── Enhanced Helper Functions ──────────────────────────────────────
# ----------------------------------------------------------------------
def robust_zscore(arr: np.ndarray, min_std: float = 1e-8) -> np.ndarray:
    """
    Compute z-score with robust handling of edge cases.
    
    Args:
        arr: Input array
        min_std: Minimum standard deviation to avoid division by zero
    """
    if len(arr) == 0:
        return arr
        
    # Handle NaN values
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return arr
    
    mu = np.nanmean(arr)
    sigma = np.nanstd(arr)
    
    # Avoid division by very small numbers
    if sigma < min_std:
        log.debug(f"Low variance detected (σ={sigma:.2e}), using mean centering only")
        return arr - mu
    
    return (arr - mu) / sigma

def compute_similarity_score(dtw_distance: float, max_expected_dtw: float = 50.0) -> float:
    """
    Convert DTW distance to similarity score (0-1, higher = more similar).
    This makes DTW comparable to cointegration similarity scores.
    """
    if not np.isfinite(dtw_distance) or dtw_distance < 0:
        return 0.0
    
    # Use exponential decay to convert distance to similarity
    # Score approaches 1 as distance approaches 0
    similarity = np.exp(-dtw_distance / max_expected_dtw)
    return min(1.0, max(0.0, similarity))

def compute_dtw_with_stats(
    a: pd.Series, 
    b: pd.Series, 
    dtw_window: Optional[int] = None
) -> Tuple[float, float, float, float]:
    """
    Compute DTW distance along with additional statistics.
    
    Returns:
        Tuple of (dtw_distance, dtw_normalized, correlation, volatility_ratio)
    """
    # Ensure series are aligned and clean
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 2:
        return float('inf'), float('inf'), 0.0, 0.0
    
    series_a, series_b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    
    # Z-score normalization
    norm_a = robust_zscore(series_a.values)
    norm_b = robust_zscore(series_b.values)
    
    try:
        # DTW calculation with error handling
        dtw_dist = float(dtw.distance(
            norm_a, norm_b, 
            window=dtw_window, 
            use_pruning=True
        ))
        
        # Normalize DTW by sequence length
        dtw_normalized = dtw_dist / len(norm_a) if len(norm_a) > 0 else float('inf')
        
    except Exception as e:
        log.warning(f"DTW calculation failed: {e}")
        return float('inf'), float('inf'), 0.0, 0.0
    
    # Additional statistics
    correlation = series_a.corr(series_b) if len(series_a) > 1 else 0.0
    vol_a, vol_b = series_a.std(), series_b.std()
    volatility_ratio = vol_a / vol_b if vol_b > 0 else 0.0
    
    return dtw_dist, dtw_normalized, correlation, volatility_ratio

# ----------------------------------------------------------------------
# 4 ─── Data Loading ───────────────────────────────────────────────────
# ----------------------------------------------------------------------
def load_interval(repo_root: Path, cryptos: List[str], interval: str) -> pd.DataFrame:
    """Load data for a specific interval with validation."""
    dfs, missing = [], []
    interval_path = repo_root / "data" / "processed" / interval
    
    if not interval_path.exists():
        raise FileNotFoundError(f"Interval directory not found: {interval_path}")
    
    for sym in cryptos:
        file_path = interval_path / f"{sym}.parquet"
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                if 'close' not in df.columns:
                    log.warning(f"No 'close' column in {file_path}")
                    missing.append(sym)
                    continue
                
                # Validate data quality
                close_data = df[["close"]].rename(columns={"close": sym})
                if close_data.empty or close_data.isna().all().iloc[0]:
                    log.warning(f"Empty or all-NaN data in {file_path}")
                    missing.append(sym)
                    continue
                
                dfs.append(close_data)
                
            except Exception as e:
                log.error(f"Error loading {file_path}: {e}")
                missing.append(sym)
        else:
            missing.append(sym)
    
    if missing:
        log.warning(f"[{interval}] Missing {len(missing)}/{len(cryptos)} symbols: {', '.join(missing)}")
    
    if not dfs:
        raise FileNotFoundError(f"No valid data files found for interval {interval}")
    
    # Merge and validate
    merged = pd.concat(dfs, axis=1).sort_index()
    
    # Remove completely empty rows
    merged = merged.dropna(how='all')
    
    if merged.empty:
        raise ValueError(f"No valid data after cleaning for interval {interval}")
    
    log.info(f"[{interval}] Loaded {merged.shape[1]} symbols × {merged.shape[0]} rows")
    return merged

# ----------------------------------------------------------------------
# 5 ─── Window Generation ──────────────────────────────────────────────
# ----------------------------------------------------------------------
def generate_windows(
    start_date: str, 
    end_date: str, 
    window_days: int, 
    step_days: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate sliding windows with validation."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    if start >= end:
        raise ValueError(f"Start date {start_date} must be before end date {end_date}")
    
    if window_days <= 0 or step_days <= 0:
        raise ValueError("Window and step days must be positive")
    
    windows = []
    current = start
    
    while current + pd.Timedelta(days=window_days - 1) <= end:
        window_end = current + pd.Timedelta(days=window_days - 1)
        windows.append((current, window_end))
        current += pd.Timedelta(days=step_days)
    
    log.info(f"Generated {len(windows)} windows from {start_date} to {end_date}")
    return windows

# ----------------------------------------------------------------------
# 6 ─── Main Processing Logic ──────────────────────────────────────────
# ----------------------------------------------------------------------
def analyze_interval(config: Dict[str, Any], repo_root: Path, interval: str) -> Optional[Path]:
    """Analyze a single interval with standardized output format."""
    log.info(f"▶ Analyzing interval {interval}...")
    start_time = time.time()
    
    try:
        data = load_interval(repo_root, [c.lower() for c in config["cryptos"]], interval)
    except (FileNotFoundError, ValueError) as e:
        log.error(f"Failed to load data for {interval}: {e}")
        return None
    
    results = []
    stats = {"tested": 0, "too_short": 0, "failed": 0, "similar": 0}
    
    # Generate all crypto pairs
    crypto_pairs = list(combinations([c.lower() for c in config["cryptos"]], 2))
    log.info(f"Processing {len(crypto_pairs)} crypto pairs across windows")
    
    # Generate windows
    windows = generate_windows(
        config["start_date"],
        config["end_date"],
        config["window_days"],
        config["step_days"]
    )
    
    for w_start, w_end in windows:
        window_slice = data.loc[w_start:w_end]
        
        for a, b in crypto_pairs:
            if a not in window_slice.columns or b not in window_slice.columns:
                continue
            
            pair_df = window_slice[[a, b]].dropna()
            
            # Need enough overlapping observations
            if len(pair_df) < config["min_rows_per_window"]:
                stats["too_short"] += 1
                continue
            
            stats["tested"] += 1
            tag = f"{a.upper()}-{b.upper()} {w_start.date()}→{w_end.date()}"
            
            # Compute DTW and statistics
            dtw_dist, dtw_norm, correlation, vol_ratio = compute_dtw_with_stats(
                pair_df[a], 
                pair_df[b], 
                config.get("dtw_window_size")
            )
            
            # Skip if DTW calculation failed
            if not np.isfinite(dtw_dist):
                stats["failed"] += 1
                log.debug(f"DTW calculation failed for {tag}")
                continue
            
            # Compute similarity score
            similarity = compute_similarity_score(dtw_norm, config.get("dtw_threshold", 10.0))
            
            # Classify as similar if DTW distance is below threshold
            is_similar = dtw_norm < config.get("dtw_threshold", 10.0)
            if is_similar:
                stats["similar"] += 1
            
            # Standardized output format (matching cointegration structure)
            results.append({
                "method": "dtw",
                "interval": interval,
                "window_start": w_start.date().isoformat(),
                "window_end": w_end.date().isoformat(),
                "asset1": a.upper(),
                "asset2": b.upper(),
                "rows": len(pair_df),
                
                # DTW-specific metrics
                "dtw_distance": round(dtw_dist, 6),
                "dtw_normalized": round(dtw_norm, 6),
                "similar": is_similar,
                "dtw_threshold": config.get("dtw_threshold", 10.0),
                
                # Common metrics (comparable with cointegration)
                "correlation": round(correlation, 4),
                "volatility_ratio": round(vol_ratio, 4),
                "similarity_score": round(similarity, 6),
            })
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        output_dir = repo_root / "reports" / "dtw"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"dtw_{interval}.csv"
        df.to_csv(output_path, index=False)
        
        processing_time = time.time() - start_time
        log.info(
            f"[{interval}] Completed: {stats['tested']} pairs tested, "
            f"{stats['similar']} similar, {stats['too_short']} skipped, "
            f"{stats['failed']} failed in {processing_time:.2f}s → {output_path}"
        )
        return output_path
    else:
        log.warning(f"[{interval}] No valid results generated")
        return None

# ----------------------------------------------------------------------
# 7 ─── Entry Point ────────────────────────────────────────────────────
# ----------------------------------------------------------------------
def main():
    """Main entry point with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="DTW analysis for cryptocurrency pairs with standardized output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dtw_analysis.py                    # Use default config
  python dtw_analysis.py -c custom.json    # Use custom config
  python dtw_analysis.py -v                # Enable verbose logging
        """
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config.json file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        config, repo_root = load_config(args.config)
        
        # Validate DTW-specific config
        if config.get("dtw_window_size") is not None:
            if not isinstance(config["dtw_window_size"], int) or config["dtw_window_size"] <= 0:
                raise ValueError("dtw_window_size must be a positive integer or null")
        
        # Run analysis
        result_paths = []
        for interval in config["intervals"]:
            result_path = analyze_interval(config, repo_root, interval)
            if result_path:
                result_paths.append(result_path)
        
        if result_paths:
            log.info("DTW analysis completed successfully")
            for path in result_paths:
                log.info(f"  → {path}")
            return 0
        else:
            log.error("No output files generated")
            return 1
            
    except KeyboardInterrupt:
        log.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        log.error(f"DTW analysis failed: {e}")
        if args.verbose:
            log.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())