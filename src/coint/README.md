# Cointegration Module Documentation

This module implements a comprehensive cointegration-based statistical arbitrage system for cryptocurrency pairs trading. The system uses walk-forward analysis with forming and trading periods to identify and trade cointegrated pairs.

## Module Overview

The cointegration module consists of 8 main files that work together to implement a complete statistical arbitrage system:

1. **`main_backtester.py`** - Entry point and CLI interface
2. **`config_defaults.py`** - Default configuration parameters
3. **`common_analysis.py`** - Shared utilities and data processing
4. **`statistical_tests.py`** - Cointegration and statistical testing
5. **`pair_screening.py`** - Pair identification and filtering
6. **`pair_trader.py`** - Individual pair trading logic
7. **`backtester_engine.py`** - Main backtesting orchestration
8. **`portfolio_manager.py`** - Portfolio tracking and performance

## File-by-File Breakdown

### 1. `main_backtester.py` - Entry Point
**Purpose**: Command-line interface and main execution entry point

**Key Components**:
- `main()`: Parses command-line arguments and orchestrates the backtest
- Argument parsing for interval selection (1m, 5m, 1h)
- Configuration loading and validation
- Error handling and logging setup

**How it works**:
1. Parses CLI arguments using `argparse`
2. Loads configuration from JSON files
3. Merges with default configuration values
4. Initializes the `CointegrationBacktester` engine
5. Executes the backtest and handles results

**Usage**:
```bash
python main_backtester.py --interval 1h --config config.json --verbose
```

### 2. `config_defaults.py` - Configuration Management
**Purpose**: Centralized default configuration parameters

**Key Parameters**:
- **Trading Parameters**: Entry/exit thresholds, stop-loss levels, position sizing
- **Statistical Parameters**: P-value thresholds for cointegration tests
- **Data Parameters**: Cryptocurrency symbols, date ranges, intervals
- **Risk Parameters**: Fees, slippage, maximum positions per window

**Important Defaults**:
- `entry_z: 2.0` - Z-score threshold for position entry
- `exit_z: 0.5` - Z-score threshold for position exit (0.5σ methodology compliant)
- `stop_loss_z: 3.0` - Stop-loss threshold (3.0σ)
- `taker_fee: 0.0004` - 0.04% Binance taker fee
- `slippage_rate: 0.0005` - 0.05% slippage assumption

### 3. `common_analysis.py` - Shared Utilities
**Purpose**: Common data processing, window generation, and utility functions

**Key Functions**:

#### Data Management:
- `load_interval_data()`: Loads cryptocurrency price data for specified interval
- `preprocess_series()`: Handles data preprocessing (log returns, standardization)
- `check_data_quality()`: Validates data quality and identifies issues

#### Window Generation:
- `generate_walkforward_windows_strict()`: Creates walk-forward analysis windows
- `WindowPhase`: Named tuple structure for window phases (train/trade/buffer)
- `validate_window_data_requirements()`: Ensures sufficient data for each window

#### Configuration:
- `load_config()`: Loads and validates configuration files
- `create_default_config()`: Generates default configuration
- `find_config_file()`: Locates configuration files with fallback logic

#### Analysis Utilities:
- `calculate_common_metrics()`: Computes correlation, distance, and other metrics
- `save_results()`: Saves analysis results to files
- `setup_logging()`: Standardized logging configuration

**How it works**:
1. Provides standardized data loading across different intervals
2. Implements walk-forward analysis with forming/trading/buffer periods
3. Handles data quality issues and missing values
4. Offers reusable statistical calculations

### 4. `statistical_tests.py` - Statistical Analysis
**Purpose**: Comprehensive statistical testing for cointegration analysis

**Key Classes**:

#### `CointegrationAnalyzer`:
- **`engle_granger_test()`**: Implements Engle-Granger cointegration test with multiple trend specifications
- **`johansen_test()`**: Johansen cointegration test with automatic lag selection
- **`granger_causality_test()`**: Granger causality testing for lead-lag relationships
- **`calculate_hedge_ratio()`**: Computes hedge ratios using OLS or robust methods
- **`calculate_spread_stats()`**: Calculates spread statistics (mean, std, half-life, Hurst exponent)
- **`validate_cointegration()`**: Comprehensive cointegration validation pipeline

#### Statistical Tests:
- **ADF Test**: Augmented Dickey-Fuller test for stationarity
- **Normality Tests**: Jarque-Bera and D'Agostino tests
- **Autocorrelation Tests**: Ljung-Box test for residual autocorrelation
- **Quality Scoring**: Composite quality score based on multiple metrics

**How it works**:
1. Validates input data quality and sufficiency
2. Performs multiple cointegration tests (Engle-Granger, Johansen)
3. Determines lead-lag relationships using Granger causality
4. Calculates hedge ratios and spread statistics
5. Provides comprehensive quality assessment

### 5. `pair_screening.py` - Pair Selection
**Purpose**: Identifies and filters cointegrated pairs for trading

**Key Functions**:

#### `screen_cointegrated_pairs()`:
- **Stage 0**: Data quality validation and preprocessing
- **Stage 1**: Cointegration testing and Granger causality filtering
- **Stage 2**: Pair ranking and selection based on quality metrics

#### `process_pair()`:
- Extracts pair data and validates completeness
- Applies cointegration tests using `statistical_tests.py`
- Determines lead-lag relationships
- Compiles comprehensive pair information

**Pair Information Structure**:
```python
{
    "pair": "ETHUSDT/BTCUSDT",
    "leader": "ETHUSDT",
    "follower": "BTCUSDT", 
    "hedge_ratio": 0.0456,
    "intercept": 0.0012,
    "mean": 1.0001,
    "std": 0.0234,
    "s1_vol": 0.0456,
    "s2_vol": 0.0234,
    "eg_pvalue": 0.0234,
    "johansen": True,
    "data_points": 1200,
    "coverage": 0.98
}
```

**How it works**:
1. Generates all possible cryptocurrency pairs
2. Validates data quality and completeness for each pair
3. Applies cointegration tests with multiple specifications
4. Filters pairs based on statistical significance
5. Determines lead-lag relationships using Granger causality
6. Ranks pairs by quality score and selects top candidates

### 6. `pair_trader.py` - Trading Logic
**Purpose**: Individual pair trading logic and signal generation

**Key Classes**:

#### `PairTrader`:
- **`calculate_spread()`**: Computes RATIO spread (methodology compliant)
- **`calculate_z_score()`**: Calculates z-score of current spread
- **`check_signal()`**: Generates trading signals based on Bollinger Bands strategy
- **`execute_trade()`**: Executes trades with volatility-scaled position sizing
- **`get_performance_summary()`**: Calculates pair-specific performance metrics

#### Trading Strategy:
- **Entry Conditions**: Z-score ≥ 2.0σ (long) or ≤ -2.0σ (short)
- **Exit Conditions**: |Z-score| < 0.5σ (methodology compliant)
- **Stop-Loss**: Z-score ≥ 3.0σ (long) or ≤ -3.0σ (short)
- **Time Stop**: Maximum holding period (buffer_period days)

#### Position Sizing:
- **Volatility-Based**: Inverse volatility weighting for each asset
- **Risk Management**: Base allocation divided by volatility ratios
- **Cost Calculation**: Includes fees and slippage

#### `PairTradingSignals`:
- **`bollinger_bands_signal()`**: Bollinger Bands-based signal generation
- **`mean_reversion_signal()`**: Mean reversion strategy signals

**How it works**:
1. Initializes with pair information from screening
2. Monitors spread and calculates z-scores continuously
3. Generates signals based on statistical thresholds
4. Executes trades with proper position sizing
5. Tracks performance and maintains trade history
6. Manages risk through stop-loss and time-based exits

### 7. `backtester_engine.py` - Main Engine
**Purpose**: Orchestrates the complete backtesting process

**Key Class**: `CointegrationBacktester`

#### Main Methods:
- **`load_data()`**: Loads and preprocesses price data
- **`run()`**: Executes complete walk-forward backtest
- **`execute_trading_window()`**: Processes individual trading windows

#### Walk-Forward Process:
1. **Data Loading**: Loads cryptocurrency price data for specified interval
2. **Window Generation**: Creates walk-forward windows with forming/trading periods
3. **Pair Screening**: Identifies cointegrated pairs for each forming period
4. **Trader Initialization**: Creates `PairTrader` instances for valid pairs
5. **Trading Execution**: Processes each timestamp and executes signals
6. **Portfolio Updates**: Updates portfolio value and tracks performance

#### Adaptive Parameters:
- **Minimum Rows**: Adapts based on interval (1m: 5760, 5m: 1152, 1h: 96)
- **Data Quality**: Handles missing data and ensures sufficient coverage
- **Error Handling**: Robust error handling for individual pair failures

**How it works**:
1. Loads historical price data for all cryptocurrencies
2. Generates walk-forward windows with forming/trading/buffer periods
3. For each window:
   - Screens for cointegrated pairs using forming period data
   - Initializes traders for valid pairs
   - Executes trading logic on trading period data
   - Records portfolio performance
4. Saves comprehensive results and performance metrics

### 8. `portfolio_manager.py` - Portfolio Management
**Purpose**: Tracks portfolio state, positions, and performance metrics

**Key Class**: `Portfolio`

#### Core Functions:
- **`update_position()`**: Updates asset positions after trades
- **`calculate_value()`**: Calculates current portfolio value
- **`record_value()`**: Records portfolio value at each timestamp
- **`process_trade()`**: Processes individual trades and updates portfolio
- **`calculate_performance_metrics()`**: Computes comprehensive performance metrics

#### Performance Metrics:
- **Returns**: Cumulative and annualized returns
- **Risk Metrics**: Volatility, Sharpe ratio, maximum drawdown
- **Trade Analysis**: Win rate, profit factor, average win/loss
- **Risk Management**: Stop-loss and time-stop analysis

#### Portfolio State:
- **Cash Management**: Tracks available cash
- **Position Tracking**: Maintains current positions for each asset
- **Value History**: Records portfolio value over time
- **Trade History**: Maintains complete trade records

**How it works**:
1. Initializes with starting cash balance
2. Processes each trade and updates positions accordingly
3. Calculates portfolio value at each timestamp
4. Maintains complete trade and performance history
5. Provides comprehensive performance analysis

## System Architecture

### Data Flow:
```
1. Data Loading (common_analysis.py)
   ↓
2. Window Generation (common_analysis.py)
   ↓
3. Pair Screening (pair_screening.py + statistical_tests.py)
   ↓
4. Trader Initialization (pair_trader.py)
   ↓
5. Trading Execution (backtester_engine.py + pair_trader.py)
   ↓
6. Portfolio Updates (portfolio_manager.py)
   ↓
7. Results Analysis (all modules)
```

### Key Interactions:

1. **Configuration**: `config_defaults.py` → All modules
2. **Data Processing**: `common_analysis.py` → All modules
3. **Statistical Testing**: `statistical_tests.py` → `pair_screening.py`
4. **Pair Selection**: `pair_screening.py` → `backtester_engine.py`
5. **Trading Logic**: `pair_trader.py` → `backtester_engine.py`
6. **Portfolio Management**: `portfolio_manager.py` → `backtester_engine.py`

### Walk-Forward Analysis Structure:

```
Forming Period (5 days) → Trading Period (1 day) → Buffer Period (5 days)
     ↓                        ↓                        ↓
Pair Screening          Signal Generation         Position Management
Statistical Tests       Trade Execution          Risk Management
Trader Creation         Portfolio Updates        Performance Tracking
```

## Detailed Trading Strategy & Metrics

### Core Trading Strategy: Bollinger Bands Mean Reversion

The system implements a **Bollinger Bands-based mean reversion strategy** with the following key components:

#### 1. Spread Calculation (RATIO Method)
```python
spread = leader_price / (hedge_ratio * follower_price)
```
- **Method**: Uses price ratio rather than difference (methodology compliant)
- **Hedge Ratio**: Calculated using OLS regression during forming period
- **Normalization**: Spread is normalized to have mean ≈ 1.0

#### 2. Z-Score Calculation
```python
z_score = (spread - spread_mean) / spread_std
```
- **Mean**: Calculated from forming period data
- **Standard Deviation**: Calculated from forming period data
- **Real-time**: Z-score calculated for each new price observation

### Entry Conditions

#### Long Position (Buy Spread)
- **Trigger**: `z_score <= -2.0σ`
- **Action**: Buy leader, Sell follower
- **Logic**: Spread is significantly below mean, expect reversion up

#### Short Position (Sell Spread)
- **Trigger**: `z_score >= 2.0σ`
- **Action**: Sell leader, Buy follower
- **Logic**: Spread is significantly above mean, expect reversion down

### Exit Conditions

#### Normal Exit (Methodology Compliant)
- **Trigger**: `|z_score| < 0.5σ`
- **Action**: Close position (reverse entry trades)
- **Logic**: Spread has reverted to near mean, take profit

#### Stop-Loss Exit
- **Trigger**: 
  - Long position: `z_score <= -3.0σ`
  - Short position: `z_score >= 3.0σ`
- **Action**: Close position immediately
- **Logic**: Spread continues moving against position

#### Time-Based Exit
- **Trigger**: Holding period >= `buffer_period` (5 days)
- **Action**: Close position regardless of z-score
- **Logic**: Prevent long-term exposure to deteriorating relationships

### Position Sizing & Risk Management

#### Volatility-Based Position Sizing
```python
vol_weights = {
    leader: 1 / leader_volatility,
    follower: 1 / follower_volatility
}
total_weight = sum(vol_weights.values())
leader_alloc = base_allocation * vol_weights[leader] / total_weight
follower_alloc = base_allocation * vol_weights[follower] / total_weight
```

#### Risk Parameters
- **Base Allocation**: $1,000 per pair
- **Volatility Scaling**: Inverse volatility weighting
- **Maximum Pairs**: 5 pairs per window
- **Transaction Costs**: 0.04% taker fee + 0.05% slippage

### Statistical Testing & Pair Selection

#### Cointegration Tests
1. **Engle-Granger Test**
   - **Threshold**: p-value < 0.05
   - **Method**: Multiple trend specifications (c, ct, ctt, nc)
   - **Best Result**: Lowest p-value across specifications

2. **Johansen Test**
   - **Confidence Level**: 95%
   - **Criteria**: Both trace and max eigenvalue tests
   - **Result**: Must show cointegration at 95% confidence

#### Quality Filters
- **Minimum Spread Std**: 0.01 (1% volatility)
- **Maximum Spread Std**: 1.0 (100% volatility)
- **Half-Life Range**: 1-252 days
- **Minimum Data Points**: 100 observations per window

#### Lead-Lag Analysis
- **Granger Causality**: 4 lags, p-value < 0.05
- **Cross-Correlation**: Maximum lag analysis
- **Result**: Determines leader/follower relationship

### Performance Metrics

#### Portfolio Metrics
- **Cumulative Return**: Total portfolio return
- **Annualized Volatility**: Standard deviation of returns × √252
- **Sharpe Ratio**: Return / Volatility (assuming 0% risk-free rate)
- **Maximum Drawdown**: Largest peak-to-trough decline

#### Trade Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Average win / Average loss
- **Average Win**: Mean profit of winning trades
- **Average Loss**: Mean loss of losing trades
- **Stop-Loss Count**: Number of stop-loss exits
- **Time-Stop Count**: Number of time-based exits

#### Pair-Specific Metrics
- **Hedge Ratio**: OLS coefficient from regression
- **R-Squared**: Goodness of fit for hedge ratio
- **Half-Life**: Mean reversion speed (days)
- **Hurst Exponent**: Long-memory properties (0.5 = random walk)
- **Quality Score**: Composite score (0-100) based on multiple factors

### Walk-Forward Analysis Structure

#### Window Phases
1. **Forming Period** (5 days)
   - Pair screening and selection
   - Statistical testing and validation
   - Parameter estimation (hedge ratios, means, stds)

2. **Trading Period** (1 day)
   - Signal generation and trade execution
   - Position management and monitoring
   - Performance tracking

3. **Buffer Period** (5 days)
   - Position unwinding and cleanup
   - Risk management and stop-loss monitoring
   - Preparation for next window

#### Adaptive Parameters
- **Minimum Rows**: 
  - 1m: 5,760 observations (5 days × 80%)
  - 5m: 1,152 observations (5 days × 80%)
  - 1h: 96 observations (5 days × 80%)

### Configuration Parameters

#### Trading Parameters
```python
{
    "entry_z": 2.0,              # Entry threshold (2.0σ)
    "exit_z": 0.5,               # Exit threshold (0.5σ)
    "stop_loss_z": 3.0,          # Stop-loss threshold (3.0σ)
    "buffer_period": 5,           # Maximum holding period (days)
    "base_allocation": 1000,      # Base dollar allocation
    "max_pairs_per_window": 5,    # Maximum pairs per window
}
```

#### Statistical Parameters
```python
{
    "eg_p_threshold": 0.05,      # Engle-Granger p-value threshold
    "johansen_conf_level": 0.95,  # Johansen confidence level
    "gc_lags": 4,                # Granger causality lags
    "gc_p_threshold": 0.05,      # Granger causality p-value
    "min_spread_std": 0.01,      # Minimum spread volatility
}
```

#### Cost Parameters
```python
{
    "taker_fee": 0.0004,         # 0.04% Binance taker fee
    "slippage_rate": 0.0005,     # 0.05% slippage assumption
}
```

### Methodology Compliance

The system implements methodology-compliant features:

1. **RATIO Spread**: Uses price ratio rather than difference spread
2. **0.5σ Exit**: Exits positions when |z-score| < 0.5σ
3. **Volatility Scaling**: Position sizing based on inverse volatility
4. **Walk-Forward**: Proper out-of-sample testing
5. **Risk Management**: Stop-loss and time-based exits
6. **Transaction Costs**: Realistic fee and slippage modeling

## Usage Examples

### Basic Backtest:
```python
from main_backtester import main
import sys

# Run backtest for 1-hour data
sys.argv = ['main_backtester.py', '--interval', '1h', '--verbose']
main()
```

### Custom Configuration:
```python
from backtester_engine import CointegrationBacktester
from common_analysis import load_config, setup_logging

config, repo_root = load_config('custom_config.json')
log = setup_logging('custom_backtest', verbose=True)
backtester = CointegrationBacktester(config, repo_root, '5m', log)
backtester.run()
```

### Pair Analysis:
```python
from statistical_tests import CointegrationAnalyzer
from common_analysis import load_interval_data

# Load data
data = load_interval_data(repo_root, ['ETHUSDT', 'BTCUSDT'], '1h', log)
s1, s2 = data['ETHUSDT'], data['BTCUSDT']

# Analyze cointegration
analyzer = CointegrationAnalyzer(config, log)
result = analyzer.validate_cointegration(s1, s2)
```

This modular architecture provides a robust, scalable framework for cointegration-based statistical arbitrage with comprehensive risk management and performance analysis capabilities. 