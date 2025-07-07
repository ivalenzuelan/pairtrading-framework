# Enhancing Cryptocurrency Pairs Trading  
### An Integrated Framework Combining Cointegration and Causal Inference

> **Summary:** This research introduces a novel two-stage framework integrating cointegration screening with Granger-causality filtering for cryptocurrency pairs trading. The methodology demonstrates significant improvements over traditional approaches, with a 61.7% return and 3.18 Sharpe ratio at 5-minute intervals. The modular Python implementation enables robust backtesting and parameter optimization.

---

## Key Innovations
- **Causal Enhancement:** First formal integration of Granger-causality filtering with cointegration (+14.8 pp returns)
- **Volatility-Adaptive Sizing:** Position scaling ∝ 1/σᵢ reduces drawdowns by 29-32%
- **Latency-Constrained Execution:** Actionable 1-4 bar lag window
- **Modular Architecture:** 8-component Python system for walk-forward validation

---

## Methodology
### Two-Stage Framework
1. **Cointegration Screening**  
   - Johansen trace test (p < 0.05)  
   - Spread: Zᵢⱼₜ = Pᵢₜ/(βPⱼₜ)  
   - Validation: Half-life (1-252 days), σz > 0.01

2. **Granger-Causality Filtering**  
   - VECM: ΔYₜ = αβ'Yₜ₋₁ + ΣΓᵢΔYₜ₋ᵢ + εₜ  
   - Unidirectional causality (Wald test, p < 0.05)  
   - Practical latency: 1-4 bars

### Trading Engine
| Component       | Specification               |
|-----------------|-----------------------------|
| Entry Trigger   | \|Z\| > 2.0σ               |
| Exit Trigger    | \|Z\| < 0.5σ               |
| Stop-Loss       | \|Z\| > 3.0σ OR 5-day hold |
| Costs           | 4 bps fee + 5 bps slippage |

---

## System Architecture
![Framework Diagram](https://raw.githubusercontent.com/ivalenzuelan/pairtrading-framework/images/flow.png)  
*Modular implementation with data flow*

---

## Next Steps
1. Extended backtesting (2019-2024)  
2. Regime-switching model integration  
3. Live paper trading implementation  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/ivalenzuelan/pairtrading-framework)
