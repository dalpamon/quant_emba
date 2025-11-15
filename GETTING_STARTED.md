# Your Personalized Finance Project Setup Guide

## What You're Building
You're implementing a **quantitative trading strategy** (likely momentum) using real stock market data for your final team presentation.

---

## Current Status

### ✅ Already Done
- Python 3.12.3 installed
- WRDS library installed
- Project folder created at `/mnt/c/Users/iamsu/CascadeProjects/quant1`

### 🔄 Next Steps
1. Get WRDS account access
2. Install remaining Python libraries
3. Download stock market data
4. Implement trading strategy
5. Prepare presentation

---

## Step 1: Get WRDS Access (CRITICAL - DO THIS FIRST)

### What is WRDS?
A database with stock prices and company financial data from 1960s to today.

### How to Get Access

1. **Go to:** https://wrds-www.wharton.upenn.edu/
2. **Click:** "Register" or "Create Account"
3. **Use:** Your SNU email (@snu.ac.kr)
4. **Wait:** 1-2 business days for approval
5. **If stuck:** Ask Professor Lee (kuanlee@snu.ac.kr) or contact WRDS support

**⚠️ Start this TODAY - approval can take time!**

---

## Step 2: Install Required Python Libraries

You already have `wrds`, but you need more libraries for data analysis:

```bash
# In your WSL terminal, run:
pip install pandas numpy matplotlib seaborn statsmodels scipy sqlalchemy psycopg2-binary --break-system-packages
```

### What each library does:
- **pandas**: Manipulate data (like Excel but in Python)
- **numpy**: Math calculations
- **matplotlib/seaborn**: Create charts
- **statsmodels/scipy**: Statistical tests
- **sqlalchemy/psycopg2-binary**: Connect to WRDS database

---

## Step 3: Test WRDS Connection (Once You Have Account)

Create a test file to verify everything works:

```python
# test_wrds.py
import wrds
import pandas as pd

# Connect to WRDS (will ask for username/password)
db = wrds.Connection()

# See what's available
print("Available libraries:")
print(db.list_libraries())

# Close connection
db.close()
print("\n✅ Connection successful!")
```

**Run it:**
```bash
python3 test_wrds.py
```

---

## Step 4: Understand the Data

### CRSP Data (Stock Prices)
This is what you'll mainly use:

| Column | What It Means | Example |
|--------|---------------|---------|
| `date` | Month end date | 2024-01-31 |
| `permno` | Stock ID number | 10001 |
| `ret` | Monthly return | 0.05 = 5% gain |
| `prc` | Stock price | -45.25 (use abs value) |
| `shrout` | Shares outstanding | 3680 (in thousands) |

**Key Formula:**
- Market Cap = abs(prc) × shrout × 1000

---

## Step 5: Your Project - Momentum Strategy

### What is Momentum?
**Simple idea:** Stocks that went up in the past tend to keep going up (at least for a while).

### The Strategy (Simplified)

```
Every month:
1. Look at all stocks
2. Calculate: How much did each stock gain/lose in past 12 months?
3. Buy the top 10% performers (winners)
4. Sell the bottom 10% performers (losers)
5. Hold for 1 month
6. Repeat

Question: Do winners beat losers?
```

### Implementation Steps

**Step 1: Download Data**
```python
import wrds
import pandas as pd

db = wrds.Connection()

# Get monthly stock returns from 2000-2024
query = """
    SELECT date, permno, ret, prc, shrout
    FROM crsp.msf
    WHERE date BETWEEN '2000-01-01' AND '2024-12-31'
    AND ret IS NOT NULL
"""

data = db.raw_sql(query)
data.to_csv('data/crsp_monthly.csv', index=False)
db.close()
```

**Step 2: Calculate Past Returns**
```python
import pandas as pd

df = pd.read_csv('data/crsp_monthly.csv')
df['date'] = pd.to_datetime(df['date'])

# Sort by stock and date
df = df.sort_values(['permno', 'date'])

# Calculate 12-month past return (skip most recent month)
df['past_12m_ret'] = df.groupby('permno')['ret'].rolling(12).sum().reset_index(0, drop=True).shift(1)
```

**Step 3: Form Portfolios**
```python
# For each month, rank stocks
df['rank'] = df.groupby('date')['past_12m_ret'].rank(pct=True)

# Create portfolio labels
df['portfolio'] = pd.cut(df['rank'],
                         bins=[0, 0.1, 0.9, 1.0],
                         labels=['Losers', 'Middle', 'Winners'])

# Calculate average return for each portfolio
results = df.groupby(['date', 'portfolio'])['ret'].mean().unstack()
```

**Step 4: Test & Visualize**
```python
import matplotlib.pyplot as plt

# Average returns
avg_returns = results.mean() * 100  # Convert to percentage
print("\nAverage Monthly Returns:")
print(avg_returns)

# Plot cumulative returns
(1 + results).cumprod().plot(figsize=(12, 6))
plt.title('Momentum Strategy: Winners vs Losers')
plt.ylabel('Cumulative Return')
plt.savefig('output/momentum_results.png')
plt.show()

# Statistical test
winners_losers_diff = results['Winners'] - results['Losers']
print(f"\nWinner-Loser Spread: {winners_losers_diff.mean()*100:.2f}% per month")
print(f"T-statistic: {winners_losers_diff.mean() / winners_losers_diff.std() * (len(winners_losers_diff)**0.5):.2f}")
```

---

## Step 6: Create Folder Structure

```bash
# Run this in your terminal
mkdir -p data/raw data/processed output/figures output/results code
```

Your folders:
```
quant1/
├── code/              # Your Python scripts
├── data/
│   ├── raw/          # Downloaded data
│   └── processed/    # Cleaned data
├── output/
│   ├── figures/      # Charts
│   └── results/      # Tables, statistics
└── GETTING_STARTED.md
```

---

## Quick Reference: Key Concepts

### Returns
```python
# Monthly return
ret = (price_today - price_last_month + dividends) / price_last_month

# 12-month cumulative return (simple)
total_ret = (1 + ret_month1) * (1 + ret_month2) * ... - 1
```

### Portfolio Weighting
- **Equal-weighted**: Each stock gets same amount ($)
- **Value-weighted**: Bigger companies get more weight

### Statistical Significance
- **T-statistic > 2**: Probably real pattern
- **T-statistic < 2**: Could be random luck

---

## Common Mistakes to Avoid

1. ❌ Using current month return to predict current month
   ✅ Use past 12 months (t-13 to t-2) to predict next month (t+1)

2. ❌ Forgetting to skip month t-1
   ✅ Skip it (reversal effect)

3. ❌ Not handling missing data
   ✅ Remove or fill missing values properly

4. ❌ Testing on same data you used to build strategy
   ✅ Use different time periods (in-sample vs out-of-sample)

---

## Timeline (Based on Dec 20 Presentation)

### Week 1 (This Week)
- [ ] Apply for WRDS access
- [ ] Install all Python libraries
- [ ] Set up folder structure
- [ ] Test WRDS connection

### Week 2-3
- [ ] Download CRSP data
- [ ] Explore and clean data
- [ ] Understand data structure

### Week 4-5
- [ ] Implement momentum strategy
- [ ] Calculate portfolio returns
- [ ] Run statistical tests

### Week 6-7
- [ ] Create visualizations
- [ ] Prepare presentation slides
- [ ] Practice presentation

### Week 8 (Dec 20)
- [ ] **Final presentation (500 points)**

---

## What Your Presentation Needs

1. **Strategy Explanation** (2 min)
   - What is momentum?
   - Why should it work?

2. **Data & Methodology** (3 min)
   - Data source: CRSP 2000-2024
   - Portfolio formation method
   - Rebalancing frequency

3. **Results** (5 min)
   - Average returns (Winner vs Loser)
   - Statistical significance
   - Charts showing cumulative returns
   - Comparison to market

4. **Discussion** (2 min)
   - Does momentum work?
   - Limitations
   - Real-world considerations

---

## Getting Help

1. **Professor Lee:** kuanlee@snu.ac.kr
2. **TA:** thpaseong@snu.ac.kr
3. **WRDS Support:** wrds@wharton.upenn.edu
4. **Me (Claude):** Ask me anything about the code!

---

## Important Commands Cheat Sheet

```bash
# Install library
pip install library_name --break-system-packages

# Run Python script
python3 script_name.py

# Start Python interactive
python3

# Check if library installed
python3 -c "import library_name; print('OK')"
```

---

## Next Action: DO THIS NOW

1. Go to https://wrds-www.wharton.upenn.edu/
2. Register with your @snu.ac.kr email
3. While waiting for approval, install the Python libraries above
4. Come back and tell me when you have WRDS access

Then we'll download your first dataset together! 🚀
