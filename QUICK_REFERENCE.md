# ⚡ Factor Lab - Quick Reference

**One-page command reference for Factor Lab**

---

## 🚀 Getting Started (First Time)

```bash
# Navigate to project
cd quant1

# Quick start (recommended)
./run.sh

# OR Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 setup.py
streamlit run app.py
```

---

## 💻 Daily Usage

```bash
# Activate environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows

# Run app
streamlit run app.py

# Deactivate when done
deactivate
```

---

## 🧪 Testing Modules

```bash
# Test all modules (in order)
python3 core/database_schema.py
python3 core/data_loader.py
python3 core/factors.py
python3 core/portfolio.py
python3 core/backtest.py
python3 core/analytics.py
```

---

## 📊 Available Universes

| Key | Name | Tickers |
|-----|------|---------|
| `tech` | Tech Giants | AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA |
| `faang` | FAANG | META, AAPL, AMZN, NFLX, GOOGL |
| `sp500_sample` | S&P 500 Sample | Top 10 stocks |
| `dow_sample` | Dow Jones Sample | Top 10 stocks |

---

## 📈 Factor Types

| Factor | Calculation | Academic Basis |
|--------|-------------|----------------|
| **Momentum** | `(P_t / P_t-252) - 1` | Jegadeesh & Titman 1993 |
| **Value** | `1 / (Price / Book)` | Fama & French 1993 |
| **Quality** | `ROE or ROA` | Novy-Marx 2013 |
| **Size** | `-log(Market Cap)` | Fama & French 1993 |
| **Volatility** | `-std(returns)` | Ang et al. 2006 |

---

## 🎯 Performance Metrics

| Metric | Good Value | Formula |
|--------|------------|---------|
| **Sharpe Ratio** | > 1.0 | `(Return - RFR) / Volatility` |
| **Sortino Ratio** | > 1.5 | `(Return - RFR) / Downside Vol` |
| **Calmar Ratio** | > 0.5 | `CAGR / |Max Drawdown|` |
| **Max Drawdown** | < -20% | `Min((Value - Peak) / Peak)` |

---

## 🗄️ Database Tables

All tables use `quant1_` prefix:

```sql
quant1_universe         -- Stock metadata
quant1_prices           -- OHLCV data
quant1_fundamentals     -- Financial ratios
quant1_factors          -- Factor scores
quant1_backtest_runs    -- Strategy configs
quant1_backtest_results -- Equity curves
quant1_positions        -- Holdings
quant1_performance      -- Metrics
```

---

## 🔍 Useful SQL Queries

```sql
-- Check available stocks
SELECT * FROM quant1_universe;

-- Check cached price data
SELECT ticker, MIN(date), MAX(date), COUNT(*)
FROM quant1_prices
GROUP BY ticker;

-- View latest factors
SELECT * FROM quant1_factors
ORDER BY date DESC
LIMIT 10;

-- List all backtests
SELECT * FROM quant1_backtest_runs;
```

---

## 🛠️ Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| Module not found | `source venv/bin/activate` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| Data download fails | Reduce date range or use cached data |
| Database corrupt | `rm quant1_data.db && python3 setup.py` |

---

## 📁 Project Structure

```
quant1/
├── app.py              # Main Streamlit app
├── setup.py            # Database setup
├── run.sh              # Quick start script
├── requirements.txt    # Dependencies
│
├── core/               # Core modules
│   ├── database_schema.py
│   ├── data_loader.py
│   ├── factors.py
│   ├── portfolio.py
│   ├── backtest.py
│   └── analytics.py
│
├── .streamlit/         # Config
│   └── config.toml
│
└── data/               # Data storage
    └── quant1_data.db
```

---

## ⌨️ Keyboard Shortcuts (Streamlit)

| Key | Action |
|-----|--------|
| `r` | Refresh app |
| `c` | Clear cache |
| `Cmd/Ctrl + Enter` | Run cell |

---

## 🌐 URLs

| Service | URL |
|---------|-----|
| Local app | http://localhost:8501 |
| Streamlit docs | https://docs.streamlit.io |
| Streamlit Cloud | https://share.streamlit.io |

---

## 📦 Dependencies

```
streamlit 1.27.0       # Web framework
pandas 2.1.0           # Data manipulation
numpy 1.24.0           # Numerical computing
yfinance 0.2.28        # Stock data
plotly 5.17.0          # Interactive charts
scipy 1.11.0           # Statistics
matplotlib 3.8.0       # Plotting
scikit-learn 1.3.0     # ML utilities
```

---

## 🎓 Academic References

**Must-Read Papers:**

1. Fama & French (1993) - Three-Factor Model
2. Jegadeesh & Titman (1993) - Momentum
3. Novy-Marx (2013) - Quality Factor

**Recommended Books:**

- "Quantitative Momentum" - Wesley Gray
- "Factor Investing" - Andrew Berkin

---

## 💡 Pro Tips

**Speed up backtests:**
1. Use cached data (2nd run is faster)
2. Start with small universes
3. Test 1-2 years first
4. Use quarterly rebalancing

**Best practices:**
1. Always validate results manually
2. Compare to benchmark
3. Check for data issues
4. Test multiple periods
5. Consider transaction costs

---

## 🚀 Example Workflow

```bash
# 1. Start app
./run.sh

# 2. In browser:
#    - Go to Strategy Builder
#    - Select "tech" universe
#    - Set dates: 2020-01-01 to 2024-10-25
#    - Momentum: 50%, Value: 50%
#    - Click "Run Backtest"

# 3. View results
#    - Check Sharpe ratio > 1.0
#    - Review drawdown periods
#    - Export if satisfied

# 4. Iterate
#    - Try different factor weights
#    - Test different universes
#    - Compare strategies
```

---

## 📝 Quick Customization

### Add a new universe:

Edit `core/data_loader.py`:

```python
UNIVERSES = {
    'my_universe': {
        'name': 'My Custom Universe',
        'tickers': ['TICKER1', 'TICKER2', ...]
    }
}
```

### Adjust transaction costs:

In Strategy Builder sidebar:
- Transaction Cost: 10 bps (default)
- Increase for conservative estimates

---

## 🎯 Success Metrics

**Good backtest results:**
- Total Return: > S&P 500
- Sharpe Ratio: > 1.0
- Max Drawdown: < -25%
- Win Rate: > 50%
- Profit Factor: > 1.5

---

## 📞 Help

**Read in order:**
1. `README.md` - Overview
2. `INSTALLATION.md` - Setup guide
3. `IMPLEMENTATION_SUMMARY.md` - Technical details
4. Module docstrings - API docs

---

**Last Updated**: October 25, 2024
**Version**: 1.0
