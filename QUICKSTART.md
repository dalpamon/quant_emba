# ⚡ Quick Start Guide - Factor Lab

Get up and running in **5 minutes**!

---

## 📦 Step 1: Download & Setup (2 minutes)

```bash
# 1. Create project directory
mkdir factor-lab
cd factor-lab

# 2. Save all the files from Claude to this directory

# 3. Install dependencies
pip install -r requirements.txt
```

**What gets installed:**
- `pandas`, `numpy` - Data manipulation
- `yfinance` - Stock data
- `streamlit` - Web interface
- `plotly` - Interactive charts
- `scipy`, `scikit-learn` - Statistics

---

## 🚀 Step 2: Initialize Database (3 minutes)

```bash
# Quick setup (5 stocks, fast for testing)
python setup_database.py --quick

# OR full setup (20 stocks, takes 5-10 min)
python setup_database.py
```

**What this does:**
1. ✅ Creates `quant1_data.db` SQLite database
2. ✅ Downloads historical prices (2020-2024)
3. ✅ Loads fundamental data (P/E, P/B, ROE)
4. ✅ Calculates factors (momentum, value, quality)

**Output:**
```
✅ Loaded 5 stocks into quant1_universe
✅ Loaded 6,325 price records
✅ Loaded fundamentals for 5 stocks
✅ Calculated and saved 6,325 factor records

📊 DATABASE SUMMARY
quant1_universe           5 rows
quant1_prices         6,325 rows
quant1_fundamentals       5 rows
quant1_factors        6,325 rows
```

---

## 🎨 Step 3: Launch Web App (< 1 minute)

```bash
streamlit run app.py
```

Browser opens automatically at: **http://localhost:8501**

---

## 🎯 Step 4: Run Your First Backtest

**In the Streamlit app:**

1. **Left Sidebar → Strategy Builder**
   - Momentum: 40%
   - Value: 30%
   - Quality: 30%

2. **Portfolio Parameters**
   - Rebalance: Monthly
   - Long: 20% (top performers)
   - Short: 20% (bottom performers)
   - Initial Capital: $100,000

3. **Click: 🚀 Run Backtest**

**Results appear in ~10 seconds:**
- 📈 Equity Curve
- 📊 Performance Metrics
- 📉 Drawdown Analysis
- 📋 Position History

---

## 📊 Understanding Your Results

### Key Metrics

**Return Metrics:**
- **Total Return**: Did you make money? (e.g., 45% = good!)
- **CAGR**: Annualized return (e.g., 12% = solid)

**Risk Metrics:**
- **Sharpe Ratio**: Risk-adjusted return
  - \> 1.0 = Good
  - \> 2.0 = Excellent
  - \> 3.0 = Exceptional
- **Max Drawdown**: Worst loss from peak (e.g., -15% = acceptable)

**Win Rate:**
- Percentage of profitable days (55-60% is typical)

---

## 🔬 Try Different Strategies

### Strategy 1: Pure Momentum
```
Momentum: 100%
Value: 0%
Quality: 0%

Long: 30%, Short: 30%
Rebalance: Monthly
```

**Theory:** Ride the trends (Jegadeesh & Titman 1993)

### Strategy 2: Value Investing
```
Momentum: 0%
Value: 100%
Quality: 0%

Long: 20%, Short: 20%
Rebalance: Quarterly
```

**Theory:** Buy undervalued stocks (Fama & French 1992)

### Strategy 3: Balanced Multi-Factor
```
Momentum: 40%
Value: 30%
Quality: 30%

Long: 20%, Short: 20%
Rebalance: Monthly
```

**Theory:** Diversify across factors (Carhart 1997)

---

## 💡 Pro Tips

### Optimize Your Strategy

1. **Adjust Factor Weights**
   - Try different combinations
   - Compare Sharpe ratios
   - Look for consistency

2. **Test Different Rebalancing**
   - Monthly: Higher returns, higher costs
   - Quarterly: Lower costs, more stable
   - Weekly: Very active, needs low costs

3. **Vary Long/Short %**
   - 10/10: Conservative (fewer positions)
   - 20/20: Balanced (default)
   - 30/30: Aggressive (more positions)

### Analyze Results

**Good Strategy Characteristics:**
- Sharpe Ratio > 1.5
- Max Drawdown < 20%
- Consistent monthly returns
- Win rate > 55%

**Red Flags:**
- Very high Sharpe (> 5) = Overfitting
- Long recovery from drawdowns
- Erratic monthly returns
- Win rate < 50%

---

## 🐛 Troubleshooting

### Database Error
```
❌ Database not found or empty!
```

**Fix:**
```bash
python setup_database.py --quick
```

### Import Error
```
ModuleNotFoundError: No module named 'yfinance'
```

**Fix:**
```bash
pip install -r requirements.txt
```

### Slow Download
```
Downloading prices... (taking forever)
```

**Fix:**
- Use `--quick` mode (5 stocks instead of 20)
- Check internet connection
- Try again (Yahoo Finance can be slow)

### Empty Results
```
⚠️ No data returned
```

**Fix:**
- Check date range (must be within 2020-2024)
- Verify stocks are in database:
  ```python
  python query_examples.py
  ```

---

## 🎓 For Your Course Project

### Week-by-Week Plan

**Week 1 (Now):**
- ✅ Setup complete
- ✅ Understand factors
- ✅ Run 3-5 different strategies

**Week 2-3:**
- Test on different time periods
- Calculate all metrics
- Document what works/doesn't work

**Week 4:**
- Prepare presentation slides
- Create visualizations
- Practice explaining your strategy

**Week 8 (Dec 20):**
- 🎤 Present to class!

### Presentation Checklist

- [ ] Strategy description (which factors, why?)
- [ ] Backt results (equity curve, metrics)
- [ ] Risk analysis (drawdown, volatility)
- [ ] Why it works (theory + evidence)
- [ ] Limitations (transaction costs, sample size)
- [ ] Future improvements

---

## 📚 Next Steps

### Explore the Code

**Want to understand how it works?**

1. **Start with `factors.py`**
   - See how momentum, value, quality are calculated
   - Theory citations for each factor

2. **Then `portfolio.py`**
   - Long-short construction
   - Position weighting

3. **Then `backtest.py`**
   - Simulation engine
   - Transaction costs

4. **Finally `analytics.py`**
   - All the performance metrics
   - Formula explanations

### Add Custom Features

**Want to add your own factors?**

```python
# In factors.py
@staticmethod
def my_custom_factor(data):
    """Your factor logic"""
    return data  # Your calculation
```

**Want Korean market data?**
- See README.md section "Add Korean Market Data"
- Install `FinanceDataReader` and `pykrx`

### Advanced Usage

**Python API (for programmatic backtesting):**

```python
from data_loader import DataLoader
from backtest import Backtester
from analytics import PerformanceAnalytics

# Load data
loader = DataLoader('quant1_data.db')

# Run backtest
backtester = Backtester(prices, factors, 100000)
equity, positions = backtester.run(
    '2020-01-01', '2024-10-25',
    rebalance_freq='M',
    top_pct=0.2,
    bottom_pct=0.2
)

# Analyze
report = PerformanceAnalytics.generate_report(equity)
PerformanceAnalytics.print_report(report)
```

---

## 🆘 Need Help?

### Resources

1. **Read README.md** - Comprehensive documentation
2. **Check `query_examples.py`** - SQL query examples
3. **Review academic papers** - Citations in factors.py

### For Course

- **TA:** 조성우 (thpaseong@snu.ac.kr)
- **Professor:** Kuan-Hui Lee (kuanlee@snu.ac.kr)

### Common Questions

**Q: Can I use this for real trading?**
A: No! This is educational only. Real trading requires:
- Better data quality
- More robust risk management
- Proper regulatory compliance

**Q: Why do my results differ from classmates?**
A: Different factor weights, rebalancing, dates = different results. That's the point!

**Q: What Sharpe ratio should I aim for?**
A: 1.5+ is good for a course project. Don't chase perfection.

**Q: How do I explain why my strategy works?**
A: Link to academic theory:
- Momentum: Jegadeesh & Titman (1993)
- Value: Fama & French (1992)
- Quality: Novy-Marx (2013)

---

## ✅ Success Checklist

- [ ] Database setup complete
- [ ] App runs successfully
- [ ] First backtest completed
- [ ] Understand all metrics
- [ ] Tried 3+ different strategies
- [ ] Know which one to present
- [ ] Understand why it works
- [ ] Ready for Q&A

---

## 🎉 You're Ready!

You now have a professional quant platform. Go build your strategy and ace that presentation!

**Remember:** The best strategy isn't the highest return - it's the one you can explain and defend.

Good luck! 🚀

---

**Quick Commands Reference:**

```bash
# Setup
python setup_database.py --quick

# Launch app
streamlit run app.py

# Query database
python query_examples.py

# Test a module
python factors.py
python backtest.py
python analytics.py
```
