# Pair Trading Crypto Research Project

A comprehensive statistical arbitrage system for cryptocurrency pairs trading, featuring advanced cointegration analysis with Vector Error Correction Model (VECM) implementation.

## 🚀 Project Overview

This project implements a sophisticated statistical arbitrage system designed for cryptocurrency markets. It features multiple analysis methods including traditional cointegration, correlation-based approaches, distance metrics, and the newly implemented **VECM (Vector Error Correction Model)** framework for enhanced pair selection and trading signal generation.

## 📊 Key Features

### 🔬 Advanced Statistical Analysis
- **VECM Implementation**: Vector Error Correction Model for robust cointegration analysis
- **Multiple Testing Methods**: Engle-Granger, Johansen, and VECM-based tests
- **Granger Causality**: VECM-based causality testing for lead-lag relationships
- **Comprehensive Diagnostics**: Model validation and residual analysis

### 📈 Trading Strategies
- **Statistical Arbitrage**: Mean reversion strategies based on cointegration
- **Walk-Forward Analysis**: Robust out-of-sample testing methodology
- **Risk Management**: Position sizing, stop-loss, and time-based exits
- **Multi-Interval Support**: 1-minute, 5-minute, and 1-hour data intervals

### 🛠 Technical Implementation
- **Modular Architecture**: Clean separation of analysis, trading, and evaluation components
- **Comprehensive Logging**: Detailed execution tracking and debugging
- **Performance Metrics**: Sharpe ratio, drawdown analysis, and trade statistics
- **Data Pipeline**: Automated data processing and quality validation

## 🏗 Project Structure

```
pair-trading-crypto-research/
├── src/
│   ├── analysis/           # Statistical analysis modules
│   │   ├── coint_full.py  # Traditional cointegration analysis
│   │   ├── corr_full.py   # Correlation-based analysis
│   │   ├── dist_full.py   # Distance-based analysis
│   │   ├── hurst_full.py  # Hurst exponent analysis
│   │   └── sdr_full.py    # Sparse Dictionary Representation
│   ├── coint/             # VECM and cointegration implementation
│   │   ├── vecm.py        # VECM Granger causality implementation
│   │   ├── statistical_tests.py  # Enhanced statistical testing
│   │   ├── pair_screening.py     # Pair identification and filtering
│   │   ├── pair_trader.py        # Individual pair trading logic
│   │   ├── backtester_engine.py  # Main backtesting orchestration
│   │   ├── portfolio_manager.py  # Portfolio tracking and performance
│   │   └── main_backtester.py    # Entry point and CLI interface
│   ├── trading/           # Trading execution modules
│   ├── evaluation/        # Performance evaluation
│   ├── selection/         # Pair selection algorithms
│   ├── pipeline/          # Data processing pipeline
│   └── utils/             # Utility functions
├── data/
│   ├── raw/              # Raw cryptocurrency data
│   └── processed/        # Processed and cleaned data
├── reports/              # Backtest results and analysis
├── config/               # Configuration files
└── notebooks/            # Jupyter notebooks for analysis
```

## 🔧 VECM Implementation

### Overview
The VECM (Vector Error Correction Model) implementation provides a robust framework for analyzing cointegrated time series. This approach extends traditional cointegration analysis by explicitly modeling the error correction mechanism and short-run dynamics.

### Key Components

#### 1. VECM Model Fitting (`vecm.py`)
```python
class VECMGrangerCausalityMixin:
    def vecm_granger_causality_test(self, df, coint_rank=1, k_ar_diff=1, det_order=0, alpha=0.05):
        """
        Perform VECM-based Wald test for Granger causality
        
        Tests the null hypothesis that lagged differences of one variable
        do not Granger-cause another variable in the VECM framework.
        """
```

#### 2. Enhanced Statistical Testing (`statistical_tests.py`)
```python
class CointegrationAnalyzer(VECMGrangerCausalityMixin):
    def fit_vecm_model(self, data, coint_rank=None, lags=None, deterministic=None):
        """
        Fit Vector Error Correction Model with automatic parameter selection
        """
    
    def comprehensive_vecm_analysis(self, data, symbol_names=None):
        """
        Complete VECM analysis including cointegration, causality, and diagnostics
        """
```

#### 3. Model Diagnostics
- **Portmanteau Test**: Residual autocorrelation testing
- **Jarque-Bera Test**: Normality of residuals
- **Heteroscedasticity Test**: Variance stability
- **Information Criteria**: AIC, BIC for model selection

### VECM Features

#### Error Correction Mechanism
The VECM explicitly models the error correction term:
```
ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_kΔY_{t-k} + ε_t
```

Where:
- `α`: Speed of adjustment coefficients
- `β`: Cointegrating vectors
- `Γ_i`: Short-run dynamics coefficients

#### Granger Causality Testing
VECM-based Wald tests for causality in both directions:
- Tests if variable A Granger-causes variable B
- Tests if variable B Granger-causes variable A
- Provides significance levels and test statistics

#### Impulse Response Analysis
- Tracks how shocks propagate through the system
- Identifies lead-lag relationships
- Measures the persistence of effects

## 📊 Analysis Methods

### 1. Traditional Cointegration (`coint_full.py`)
- Engle-Granger test with multiple trend specifications
- Johansen cointegration test with automatic lag selection
- Hedge ratio calculation using OLS regression
- Spread statistics and half-life analysis

### 2. Correlation-Based Analysis (`corr_full.py`)
- Pearson and Spearman correlation analysis
- Rolling correlation windows
- Correlation-based pair selection
- Dynamic correlation tracking

### 3. Distance-Based Analysis (`dist_full.py`)
- Euclidean distance metrics
- Dynamic time warping (DTW)
- Distance-based similarity scoring
- Multi-dimensional distance analysis

### 4. Sparse Dictionary Representation (`sdr_full.py`)
- Dictionary learning for feature extraction
- Sparse coding for similarity measurement
- Dimensionality reduction techniques
- Non-linear relationship detection

### 5. Hurst Exponent Analysis (`hurst_full.py`)
- Long-memory process detection
- Fractal analysis for time series
- Mean reversion vs. trending identification
- Hurst-based pair selection

## 🎯 Trading Strategy

### Signal Generation
- **Entry Conditions**: Z-score ≥ 2.0σ (long) or ≤ -2.0σ (short)
- **Exit Conditions**: |Z-score| < 0.5σ (mean reversion)
- **Stop-Loss**: Z-score ≥ 3.0σ (long) or ≤ -3.0σ (short)
- **Time Stop**: Maximum holding period (buffer_period days)

### Position Sizing
- **Volatility-Based**: Inverse volatility weighting
- **Risk Management**: Base allocation divided by volatility ratios
- **Cost Calculation**: Includes fees and slippage

### Walk-Forward Analysis
- **Forming Period**: 5 days for pair identification
- **Trading Period**: 1 day for signal generation
- **Buffer Period**: 5 days maximum holding period
- **Step Size**: 1 day for window progression

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd pair-trading-crypto-research
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure the system**:
```bash
cp config/config.json.example config/config.json
# Edit config.json with your parameters
```

### Running Analysis

#### VECM Cointegration Analysis
```bash
python src/coint/main_backtester.py --interval 1h --config config/config.json --verbose
```

#### Traditional Cointegration
```bash
python src/analysis/coint_full.py --interval 1h --config config/config.json
```

#### Correlation Analysis
```bash
python src/analysis/corr_full.py --interval 1h --config config/config.json
```

#### Distance-Based Analysis
```bash
python src/analysis/dist_full.py --interval 1h --config config/config.json
```

### Configuration

Key configuration parameters in `config.json`:

```json
{
  "cryptos": ["ETHUSDT", "BTCUSDT", "BNBUSDT", ...],
  "intervals": ["1m", "5m", "1h"],
  "start_date": "2022-01-01",
  "end_date": "2022-03-31",
  "forming_period": 5,
  "trading_period": 1,
  "buffer_period": 5,
  "entry_z": 2.0,
  "exit_z": 0.5,
  "taker_fee": 0.0004,
  "slippage_rate": 0.0005
}
```

## 📈 Performance Metrics

The system calculates comprehensive performance metrics:

- **Cumulative Return**: Total strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Trade Statistics**: Average win/loss, trade frequency

## 📊 Results and Reports

Results are automatically saved to the `reports/` directory:

```
reports/
├── cointegration/
│   ├── 1h/
│   │   ├── equity_curve.csv
│   │   ├── trades.csv
│   │   └── metrics.json
│   ├── 5m/
│   └── 1m/
├── correlation/
├── distance/
└── backtest/
```

## 🔬 Advanced Features

### VECM Diagnostics
- **Model Validation**: Comprehensive diagnostic tests
- **Residual Analysis**: Autocorrelation and normality tests
- **Information Criteria**: AIC/BIC for model selection
- **Error Correction**: Speed of adjustment analysis

### Multi-Method Integration
- **Ensemble Approach**: Combine multiple analysis methods
- **Method Comparison**: Performance comparison across methods
- **Robust Selection**: Cross-validation for pair selection

### Data Quality Management
- **Missing Data Handling**: Forward/backward filling
- **Outlier Detection**: Statistical outlier identification
- **Data Validation**: Quality checks and reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Statsmodels**: For VECM implementation and statistical testing
- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical computations
- **Scikit-learn**: For machine learning components

## 📞 Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Note**: This system is designed for research and educational purposes. Past performance does not guarantee future results. Always perform thorough testing before using in live trading environments.
