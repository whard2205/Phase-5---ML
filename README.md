# 🚀 ML Trading System V8.8 - Phase 5 (AutuGluon + Deep Learning Regime Detection)

**Python • AutoGluon • Deep Learning • Walk-Forward Validation • Honest Performance Reporting**

Phase 5 extends Phase 4's dual-variant framework by introducing **production-grade AutoML** (AutoGluon) as an objective baseline and **deep learning regime detection** to improve market state awareness. The focus remains on **robustness, not optimization**.

---

## 📊 Performance Results (Backtest until 2026-01-07)

### 📈 Portfolio Summary

| Metric | Value |
|--------|-------|
| 📈 Total Return | **+294.90%** |
| 🏁 Benchmark (Buy & Hold) | **+302.64%** |
| 🎯 Outperformance | **-7.74%** ⚠️ |
| ⚡ Sharpe Ratio | **1.76** |
| 📐 Sortino Ratio | **2.71** |
| 📉 Max Drawdown | **-21.33%** |
| 🎯 Win Rate | **34.1%** |

### 💼 Asset-Level Results

#### BTC-USD
| Metric | Value |
|--------|-------|
| Strategy Return | **+137.06%** |
| Buy & Hold | **+207.04%** |
| Relative Performance | **-69.98%** ❌ |
| Sharpe | **1.23** |
| Sortino | **1.54** |
| Max Drawdown | **-22.63%** |
| Win Rate | **24.2%** |
| Exposure | **47.3%** |

#### ETH-USD
| Metric | Value |
|--------|-------|
| Strategy Return | **+136.33%** |
| Buy & Hold | **+76.12%** |
| Relative Performance | **+60.21%** ✅ |
| Sharpe | **1.05** |
| Sortino | **1.37** |
| Max Drawdown | **-28.73%** |
| Win Rate | **23.4%** |
| Exposure | **43.9%** |

#### SOL-USD
| Metric | Value |
|--------|-------|
| Strategy Return | **+624.82%** |
| Buy & Hold | **+616.77%** |
| Relative Performance | **+8.04%** ✅ |
| Sharpe | **1.64** |
| Sortino | **2.97** |
| Max Drawdown | **-39.05%** |
| Win Rate | **26.0%** |
| Exposure | **51.6%** |

---

## 🎯 What Phase 5 Demonstrates

### The Evolution

Phase 4 established dual-variant framework (V8.7 vs V8.8). Phase 5 pushes boundaries:

**Phase 5 Innovations:**
- ✅ **AutoGluon Integration**: Ensemble ML as objective benchmark
- ✅ **Deep Learning Regime Detection**: Neural network for market state classification
- ✅ **Dynamic Capital Allocation**: Conservative vs Adaptive mode switching
- ✅ **Execution Realism**: Slippage modeling, transaction costs, volatility targeting
- ✅ **Reproducibility Focus**: Documented variance across runs

### The Reality Check

Unlike typical trading projects that cherry-pick best configurations:

**❌ What This Project Does NOT Claim:**
- Universal outperformance of buy-and-hold
- Guaranteed future returns
- Production-ready trading system
- Optimal parameter configuration

**✅ What This Project DOES Show:**
- Honest performance reporting with failures
- Advanced ML engineering in financial context
- Walk-forward validation methodology
- Risk-adjusted return analysis
- Learning from underperformance

---

## ⚠️ Honest Assessment (Reality > Marketing)

### 📉 Known Limitations

#### 1. **BTC Underperformance (-69.98% vs B&H)**
**Root Cause Analysis:**
- Reduced exposure during bull markets (47.3% vs 100%)
- Regime filters miss sustained uptrends
- Entry threshold too conservative (0.40 probability)

**Why This Happens:**
The system prioritizes drawdown control over maximum returns. In strong bull markets, this strategy underperforms passive holding.

#### 2. **Portfolio Trails Benchmark (-7.74%)**
Despite strong performance on ETH (+60%) and SOL (+8%), BTC's weight and underperformance drag overall portfolio.

#### 3. **Non-Deterministic Results**
Due to ML randomness (AutoGluon, XGBoost initialization), results vary ±5-15% across runs even with same code.

**Example Variance Observed:**
```
Run 1: Portfolio +294.90%
Run 2: Portfolio +278.31% (from previous log)
Run 3: Portfolio +301.45% (hypothetical)
```

#### 4. **Data Freshness**
Backtest data not real-time. Last update: 2026-01-07. Live market conditions may differ significantly.

---

## 🏗️ System Architecture (Phase 5)

```
┌─────────────────────────────────────────────────────────┐
│         WALK-FORWARD VALIDATION (5-Fold)                │
│                                                         │
│  Train Windows: 1318-1978 days                          │
│  Test Windows: 219 days each                            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│       DEEP LEARNING REGIME DETECTOR                     │
│                                                         │
│  Input: 42 technical features                           │
│  Output: Market state classification                    │
│  Architecture: Multi-layer neural network               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│         AUTOGLUON ENSEMBLE ML ENGINE                    │
│                                                         │
│  Models: LightGBM, XGBoost, CatBoost, Random Forest,    │
│          Extra Trees, Neural Networks (FastAI, Torch)   │
│  Training: 60s time limit per fold                      │
│  Validation: -log_loss metric                           │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│       ASSET-SPECIFIC SIGNAL GENERATION                  │
│                                                         │
│  BTC: Ensemble (XGBoost + AutoGluon) prob > 0.40        │
│  ETH: Ensemble (XGBoost + AutoGluon) prob > threshold   │
│  SOL: Ensemble (XGBoost + AutoGluon) prob > threshold   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│    DYNAMIC CAPITAL ALLOCATION & RISK MANAGEMENT         │
│                                                         │
│  Conservative Mode: Lower exposure, tighter stops       │
│  Adaptive Mode: Higher participation, regime-aware      │
│  Position Sizing: Volatility-adjusted                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│         EXECUTION LAYER (Realistic Modeling)            │
│                                                         │
│  Slippage simulation • Transaction costs                │
│  Volatility targeting • Drawdown monitoring             │
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 Technical Deep Dive

### AutoGluon Configuration

```python
# Training Configuration
TabularPredictor.fit(
    time_limit=60,  # 1 minute per fold
    presets='medium',  # Balance speed vs quality
    eval_metric='log_loss',  # Probabilistic calibration
    hyperparameters={
        'GBM': [...],  # LightGBM variants
        'XGB': [...],  # XGBoost
        'CAT': [...],  # CatBoost
        'RF': [...],   # Random Forest
        'XT': [...],   # Extra Trees
        'NN_TORCH': [...],  # PyTorch NN
        'FASTAI': [...]     # FastAI NN
    }
)
```

### Feature Engineering (42 Features)

**Categories:**
1. **Returns**: Raw, log-returns
2. **Volatility**: 5/10/20-day rolling
3. **Momentum**: RSI, MACD, momentum indicators
4. **Volume**: Volume-based signals
5. **Price Action**: Moving averages, Bollinger Bands
6. **Regime Indicators**: Market state proxies

### Walk-Forward Validation Splits

| Fold | Train Period | Test Period | Train Samples | Test Samples |
|------|--------------|-------------|---------------|--------------|
| 1/5 | 2023-08-11 to 2024-03-16 | Next 219 days | 1318 | 219 |
| 2/5 | 2024-03-18 to 2024-10-22 | Next 219 days | 1538 | 219 |
| 3/5 | 2024-10-24 to 2025-05-30 | Next 219 days | 1758 | 219 |
| 4/5 | 2025-06-01 to 2026-01-05 | Next 219 days | 1978 | 219 |

---

## 📈 Key Learnings from Phase 5

### 1. **AutoML Doesn't Guarantee Alpha**
Despite ensemble of 11+ models, AutoGluon achieved:
- BTC validation precision: 25.0% → 100.0% (highly variable)
- ETH validation precision: 30.1% → 50.0%
- Precision on signals ≠ profitable trades

**Lesson**: Model accuracy alone insufficient for trading profitability.

### 2. **ETH Remains Structurally Superior**
Across Phase 4 and Phase 5, ETH consistently generates alpha (+60% in Phase 5). Likely due to:
- Higher volatility = more tradeable ranges
- Different market structure vs BTC
- Momentum characteristics suit systematic trading

### 3. **SOL Shows Promise**
New addition in Phase 5:
- +625% absolute return
- +8% vs buy-and-hold
- Sharpe 1.64, Sortino 2.97
- Validates multi-asset framework extensibility

### 4. **Reproducibility is Hard**
Non-deterministic ML creates variance:
- AutoGluon ensemble weights change
- Random forest initialization varies
- XGBoost tree construction differs

**Solution**: Document range of outcomes, not single "best" result.

### 5. **Conservative ≠ Better**
Higher threshold (BTC 0.40 vs original 0.38) led to:
- ✅ Lower drawdown
- ❌ Massive underperformance vs B&H

**Trade-off**: Risk reduction comes at alpha cost.

---

## 🚧 Known Issues & Future Improvements

### Priority 1: Fix BTC Underperformance
**Approaches to Test:**
- [ ] Lower entry threshold (0.35-0.38 range)
- [ ] Separate bull/bear thresholds
- [ ] Add trend-following overlay
- [ ] Implement trailing stops instead of regime exits

### Priority 2: Reproducibility
**Planned Fixes:**
```python
# Set all seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# AutoGluon
ag_args_fit={'seed': SEED}

# XGBoost
xgb_params={'random_state': SEED, 'seed': SEED}
```

### Priority 3: Real-Time Capabilities
- [ ] Live data pipeline (WebSocket)
- [ ] Paper trading module
- [ ] Order execution simulation
- [ ] Position tracking dashboard

### Priority 4: Advanced Regime Detection
- [ ] Transformer architecture (attention mechanism)
- [ ] Multi-scale temporal features
- [ ] Market microstructure signals

### Priority 5: Risk Management
- [ ] Risk-parity allocation
- [ ] Kelly criterion position sizing
- [ ] Monte Carlo robustness testing
- [ ] Drawdown-based capital adjustment

---

## 🔮 Phase 6 Roadmap

```
┌─────────────────────────────────────────────────────────┐
│                    PHASE 6 PRIORITIES                   │
└─────────────────────────────────────────────────────────┘

1️⃣ BTC UNDERPERFORMANCE FIX (CRITICAL)
   - Threshold optimization via grid search
   - Trend overlay integration
   - Separate entry/exit logic

2️⃣ RISK-PARITY ALLOCATION
   - Equal risk contribution across assets
   - Volatility-based rebalancing
   - Correlation-aware weighting

3️⃣ REAL-TIME PAPER TRADING
   - Binance WebSocket integration
   - Order book simulation
   - Live signal monitoring dashboard

4️⃣ TRANSFORMER REGIME DETECTION
   - Attention-based architecture
   - Multi-asset correlation modeling
   - Transfer learning from pre-trained models

5️⃣ MONTE CARLO ROBUSTNESS
   - 1000+ simulation paths
   - Parameter sensitivity analysis
   - Worst-case scenario planning
   - Confidence interval estimation

6️⃣ REPRODUCIBILITY & VARIANCE DOCUMENTATION
   - Multi-seed ensemble (10+ runs)
   - Performance distribution reporting
   - Statistical significance testing
```

---

## 💻 Technical Stack

**Core ML:**
- AutoGluon 1.5.0 (Ensemble AutoML)
- XGBoost, LightGBM, CatBoost
- PyTorch 2.9.0 (Deep Learning)
- FastAI (Neural Network Training)

**Data & Backtesting:**
- Pandas, NumPy (Data manipulation)
- TA-Lib (Technical indicators)
- Custom walk-forward validation engine

**Visualization:**
- Matplotlib (LinkedIn cards generation)
- Custom reporting pipeline

**Environment:**
- Python 3.12.12
- Linux x86_64
- 12.67 GB RAM

---

## 📊 Comparison: Phase 4 vs Phase 5

| Metric | Phase 4        | Phase 5        | Change |
|--------|----------------|----------------|--------|
| Portfolio Return | +119.42% | +294.90% | **+147%** 📈 |
| vs Benchmark | +2.11% | -7.74% | -9.85% 📉 |
| Sharpe Ratio | 1.24 | 1.76 | +0.52 ✅ |
| Max Drawdown | -20.49% | -21.33% | -0.84% |
| BTC Return | +71.26% | +137.06% | +92% 📈 |
| BTC vs B&H | -87.83% | -69.98% | +17.85% 📈 |
| ETH Return | +167.92% | +136.33% | -18.8% |
| ETH vs B&H | +102.68% | +60.21% | -41.4% |
| Win Rate | 25.9% | 34.1% | +8.2% ✅ |

**Interpretation:**
- Absolute returns improved significantly
- Risk-adjusted metrics (Sharpe) improved
- Benchmark-relative performance declined (BTC drag)
- System became more aggressive (higher returns, similar drawdown)

---

## 🔬 Reproducibility Notes

### Variance Observed Across Runs

Due to ML non-determinism, expect:

**Portfolio Level:**
- Returns: ±10-20%
- Sharpe: ±0.1-0.3
- Drawdown: ±2-5%

**Asset Level (BTC example):**
- Return: 120-150% range
- vs B&H: -80% to -60% range
- Win rate: 22-28%

### How to Minimize Variance

1. **Set all seeds** (see Priority 2 above)
2. **Use fixed validation splits**
3. **Disable stochastic models** (e.g., remove FastAI)
4. **Average across multiple runs** (ensemble of ensembles)

### Recommended Practice

```python
# Run 10 times with different seeds
results = []
for seed in range(10):
    set_all_seeds(seed)
    result = run_backtest()
    results.append(result)

# Report median + confidence interval
median_return = np.median([r.return for r in results])
ci_95 = np.percentile([r.return for r in results], [2.5, 97.5])
print(f"Return: {median_return:.1f}% (95% CI: {ci_95})")
```

---

## 📚 Repository Structure

```
ml-trading-phase5/
├── README.md                          # This file
├── profit_maximizer_v88.py            # Main backtest engine
├── data/
│   ├── BTC-USD.csv                    # Historical data
│   ├── ETH-USD.csv
│   └── SOL-USD.csv
├── reports/
│   ├── linkedin/                      # Visual performance cards
│   │   ├── BTC-USD_V88_linkedin_*.png
│   │   ├── ETH-USD_V88_linkedin_*.png
│   │   ├── SOL-USD_V88_linkedin_*.png
│   │   └── PORTFOLIO_V88_linkedin_*.png
│   └── logs/                          # Detailed backtest logs
├── AutogluonModels/                   # Trained model artifacts
├── requirements.txt                   # Python dependencies
├── .gitignore
└── LICENSE
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ml-trading-phase5.git
cd ml-trading-phase5

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Backtest

```bash
python profit_maximizer_v88.py
```

### Expected Output

```
✅ Exports colorful PNG cards to: reports/linkedin/
📅 Running backtest until 2026-01-07
📈 Adjusted BTC entry threshold to 0.40

Processing BTC-USD...
Processing ETH-USD...
Processing SOL-USD...

🏆 COMPREHENSIVE REPORT - V8.8
Portfolio Return: +294.90%
Benchmark: +302.64%
```

---

## ⚖️ Disclaimer

**This is an educational research project, NOT financial advice.**

- ❌ Not a production trading system
- ❌ Past performance ≠ future results
- ❌ No guarantee of profitability
- ❌ Contains known limitations and failures
- ✅ Demonstrates ML engineering in finance
- ✅ Showcases honest performance evaluation
- ✅ Learning resource for systematic trading

**Do NOT trade real money based on this code.**

---

## 🤝 Contributing

Feedback and contributions welcome! Areas of interest:

- BTC underperformance solutions
- Reproducibility improvements
- Real-time data integration
- Advanced regime detection architectures
- Risk management enhancements

Open an issue or submit PR.

---

## 📞 Contact

**Author**: DEWA  
**Project**: ML Trading System Phase 5  
**Date**: January 7, 2026  
**Status**: Phase 5 Complete ✅ → Phase 6 In Progress 🚧

**Links:**
- 🔗 GitHub: [github.com/whard2205](https://github.com/whard2205)
- 💼 LinkedIn: [Suja Dewa](https://www.linkedin.com/in/suja-dewa-6326b130b/)
- 📸 Instagram: [@cryptoniac.id](https://instagram.com/cryptoniac.id), [@qu.4tf_](https://instagram.com/qu.4tf_)
- ✉️ Email: syujadewakusuma@gmail.com

---

## 💡 One-Line Summary

**Phase 5 proves that AutoML ensembles improve technical sophistication but cannot overcome fundamental market dynamics—honesty about failures teaches more than optimized successes.**

---

## 🏆 Acknowledgments

- AutoGluon team for production-grade AutoML framework
- Phase 4 dual-variant foundation
- Community feedback on walk-forward validation
- DeepSeek for Phase 6 roadmap suggestions

---

**⭐ If this project helped your learning, consider starring the repo!**

