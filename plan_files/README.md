# 🧪 Factor Lab - Quantitative Investment Platform

A professional-grade multi-factor backtesting engine for quantitative trading strategies. Built for **Seoul National University EMBA** course: *Inefficient Market and Quant Investment*.

## 🎯 Project Overview

Factor Lab allows you to:
- Build and backtest **long-short factor strategies**
- Analyze **momentum, value, quality, and size factors**
- Calculate **risk-adjusted returns** (Sharpe, Sortino, Calmar ratios)
- Simulate **realistic transaction costs** and slippage
- Visualize performance with **interactive dashboards**

**Database:** All data stored with `quant1_` prefix for clean namespace management.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd factor-lab

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Database

```bash
# Full setup (20 tech stocks, 2020-2024)
python setup_database.py

# Quick setup (5 stocks for testing)
python setup_database.py --quick

# Custom setup
python setup_database.py --custom
```

This will:
- ✅ Create SQLite database with `quant1_` tables
- ✅ Download historical prices from Yahoo Finance
- ✅ Load fundamental data (P/E, P/B, ROE, etc.)
- ✅ Calculate factors (momentum, value, quality, size)

**Expected time:** 5-10 minutes for full setup

### 3. Launch Application

```bash
streamlit run app.py
```

Open browser to: **http://localhost:8501**

---

## 📊 Database Schema

All tables use `quant1_` prefix:

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `quant1_universe` | Stock universe | ticker, name, sector, market_cap |
| `quant1_prices` | Daily OHLCV data | ticker, date, adjusted_close |
| `quant1_fundamentals` | Fundamental metrics | ticker, pe_ratio, pb_ratio, roe |
| `quant1_factors` | Calculated factors | ticker, momentum_12m, value_pb, quality_roe |
| `quant1_backtest_runs` | Strategy configurations | run_id, factor_weights, rebalance_freq |
| `quant1_backtest_results` | Daily equity curves | run_id, date, portfolio_value |
| `quant1_positions` | Position history | run_id, ticker, weight, position_type |
| `quant1_performance` | Summary metrics | run_id, sharpe_ratio, max_drawdown |

---

## 🔬 Strategy Building

### Factor Universe

**Momentum Factors:**
- `momentum_12m`: 12-month return (skip last month)
- `momentum_6m`: 6-month return
- `momentum_3m`: 3-month return

**Value Factors:**
- `value_pb`: Book-to-Market ratio (1/P/B)
- `value_pe`: Earnings-to-Price ratio (1/P/E)
- `value_ps`: Sales-to-Price ratio (1/P/S)

**Quality Factors:**
- `quality_roe`: Return on Equity
- `quality_roa`: Return on Assets
- `quality_margin`: Net profit margin

**Size Factor:**
- `size_log_mcap`: Log of market capitalization

**Volatility Factor:**
- `volatility_60d`: 60-day rolling volatility

### Example Strategy

**Classic Fama-French Style:**
```python
factor_weights = {
    'momentum_12m': 0.33,
    'value_pb': 0.33,
    'size_log_mcap': 0.34  # Inverted: small cap premium
}
```

**Momentum-Quality:**
```python
factor_weights = {
    'momentum_12m': 0.50,
    'quality_roe': 0.30,
    'volatility_60d': 0.20  # Low volatility
}
```

---

## 📈 Performance Metrics

### Return Metrics
- **Total Return**: Cumulative return over period
- **CAGR**: Compound Annual Growth Rate
- **Excess Return**: Outperformance vs benchmark

### Risk Metrics
- **Volatility**: Standard deviation of returns (annualized)
- **Max Drawdown**: Largest peak-to-trough decline
- **Beta**: Market sensitivity
- **Downside Deviation**: Volatility of negative returns

### Risk-Adjusted Returns
- **Sharpe Ratio**: Return per unit of total risk
  - > 1.0 is good, > 2.0 is excellent
- **Sortino Ratio**: Return per unit of downside risk
  - Penalizes only negative volatility
- **Calmar Ratio**: CAGR / |Max Drawdown|
  - > 1.0 is strong
- **Information Ratio**: Active return / Tracking error
  - Measures consistency vs benchmark

### Win/Loss Analysis
- **Win Rate**: % of profitable days
- **Profit Factor**: Gross profit / Gross loss
- **Avg Win / Avg Loss**: Mean returns on up/down days

---

## 🛠️ File Structure

```
factor-lab/
├── app.py                      # Streamlit web application
├── database_schema.py          # Database table definitions
├── data_loader.py              # Data ingestion & management
├── factors.py                  # Factor calculation engine
├── portfolio.py                # Portfolio construction
├── backtest.py                 # Backtesting engine
├── analytics.py                # Performance analytics
├── setup_database.py           # Initial setup script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── quant1_data.db             # SQLite database (created by setup)
│
└── examples/
    └── query_examples.py       # SQL query examples
```

---

## 💻 Usage Examples

### Python API

```python
from data_loader import DataLoader
from factors import FactorEngine
from portfolio import Portfolio
from backtest import Backtester
from analytics import PerformanceAnalytics

# Load data
loader = DataLoader('quant1_data.db')

# Get factor data
factors = loader.get_factors_for_date('2024-10-25')

# Run backtest
backtester = Backtester(prices, factor_scores, initial_capital=100000)
equity_curve, positions = backtester.run(
    start_date='2020-01-01',
    end_date='2024-10-25',
    rebalance_freq='M',
    top_pct=0.2,
    bottom_pct=0.2
)

# Analyze performance
report = PerformanceAnalytics.generate_report(equity_curve)
PerformanceAnalytics.print_report(report)
```

### SQL Queries

```sql
-- Get latest factors for all stocks
SELECT ticker, momentum_12m, value_pb, quality_roe
FROM quant1_factors
WHERE date = (SELECT MAX(date) FROM quant1_factors)
ORDER BY momentum_12m DESC;

-- Compare all backtest runs
SELECT 
    br.run_name,
    p.sharpe_ratio,
    p.max_drawdown,
    p.total_return
FROM quant1_backtest_runs br
JOIN quant1_performance p ON br.run_id = p.run_id
ORDER BY p.sharpe_ratio DESC;

-- Get position history for a strategy
SELECT 
    rebalance_date,
    ticker,
    position_type,
    weight,
    factor_score
FROM quant1_positions
WHERE run_id = 1
ORDER BY rebalance_date, position_type, weight DESC;
```

---

## 🎓 Academic Foundation

This project implements strategies from:

1. **Fama & French (1992)** - Three Factor Model
   - Size, Value premiums
   
2. **Jegadeesh & Titman (1993)** - Momentum
   - 12-month momentum with 1-month skip
   
3. **Novy-Marx (2013)** - Quality Factor
   - Profitability (ROE, margins)
   
4. **Ang et al. (2006)** - Low Volatility Anomaly
   - Idiosyncratic volatility effect

5. **Carhart (1997)** - Four Factor Model
   - Adds momentum to Fama-French

---

## 🔧 Customization

### Add New Factors

```python
# factors.py
@staticmethod
def custom_factor(data: pd.Series) -> pd.Series:
    """Your custom factor logic"""
    return data  # Your calculation

# Then add to quant1_factors table
```

### Add Korean Market Data

```python
# Install Korean data sources
pip install finance-datareader pykrx

# In data_loader.py
import FinanceDataReader as fdr

# Download KOSPI stocks
kospi_stocks = fdr.StockListing('KOSPI')
```

### Change Rebalancing Logic

```python
# portfolio.py - modify Portfolio class
def create_custom_portfolio(self, date, **params):
    # Your portfolio logic here
    pass
```

---

## ⚠️ Important Notes

### Data Quality
- Yahoo Finance data is free but can have gaps
- Always validate results against known benchmarks
- Consider survivorship bias in historical analysis

### Transaction Costs
- Default: 10 bps (0.1%)
- US stocks: typically 5-20 bps
- Korean stocks: ~30 bps including taxes

### Overfitting Prevention
- Use walk-forward analysis
- Test on out-of-sample periods
- Avoid excessive parameter optimization

### Risk Management
- Max position size: typically 5-10%
- Sector limits: prevent concentration
- Turnover control: reduce transaction costs

---

## 📚 Next Steps

### For Your Course Project (Project 2)

1. **Switch to Korean market data**
   ```bash
   pip install finance-datareader pykrx
   ```

2. **Create `quant2_` tables for Korean data**
   - Modify `database_schema.py`
   - Use FnGuide data (from TA)

3. **Implement course-specific factors**
   - Governance factors
   - Earnings surprises
   - Analyst recommendations

4. **Final presentation (Week 8)**
   - Strategy description
   - Backtest results
   - Risk analysis
   - Future improvements

### For Production (Project 3)

1. **Migrate to PostgreSQL** for scalability
2. **Add real-time data feeds**
3. **Implement paper trading** (Interactive Brokers API)
4. **Build FastAPI backend** for multi-user access
5. **Add authentication** (OAuth, JWT)
6. **Deploy to cloud** (AWS, GCP)

---

## 🤝 Contributing

For course collaboration:
1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

---

## 📖 Resources

### Books
- *Quantitative Momentum* by Wesley Gray
- *Expected Returns* by Antti Ilmanen
- *Active Portfolio Management* by Grinold & Kahn

### Papers
- Fama-French Three Factor Model (1992)
- Jegadeesh-Titman Momentum (1993)
- Novy-Marx Quality Factor (2013)

### Websites
- Alpha Architect: alphaarchitect.com
- Quantopian Lectures: youtube.com/@quantopian
- QuantConnect Learn: quantconnect.com/learning

---

## 📧 Contact

**For SNU EMBA Students:**
- Teaching Assistant: 조성우 (thpaseong@snu.ac.kr)
- Professor: Kuan-Hui Lee (kuanlee@snu.ac.kr)

**Course:** Inefficient Market and Quant Investment  
**Term:** 2025 Fall, Module 2  
**Final Presentation:** Week 8 (Dec 20, 2025)

---

## ⚖️ License

This project is for educational purposes as part of Seoul National University's EMBA program.

**⚠️ Disclaimer:** Past performance does not guarantee future results. This tool is for educational purposes only and should not be used for actual trading without proper due diligence.

---

## ✅ Checklist for Course Success

- [ ] Database setup complete
- [ ] Understand all factors (momentum, value, quality)
- [ ] Run at least 3 different strategies
- [ ] Calculate all performance metrics
- [ ] Prepare presentation slides
- [ ] Document strategy rationale
- [ ] Analyze why strategy works/doesn't work
- [ ] Consider transaction costs
- [ ] Test robustness (different periods)
- [ ] Ready for Q&A on methodology

**Good luck with your project! 🚀**
