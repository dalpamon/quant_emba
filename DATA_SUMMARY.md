# FnGuide Data Summary - Sheet1 (2)

## Overview

**Data Source:** FnGuide (Korean Financial Data Provider)
**Data Type:** Daily stock prices
**File:** `quant.xlsx` → Sheet: `Sheet1 (2)`

---

## Dataset Specifications

### Dimensions
- **Companies:** 2,620 Korean stocks
- **Trading Days:** 366 days
- **Date Range:** Nov 14, 2024 → Nov 14, 2025 (1 year)
- **Total Data Points:** 958,920 price observations
- **Data Quality:** 98.27% complete (only 1.73% missing)

### Structure
```
              Col A        Col B         Col C           Col D
Row 10:      Symbol Name   삼성전자      SK하이닉스      LG에너지솔루션  ...
Row 15:      2024-11-14    49,900       173,000        422,000        ...
Row 16:      2024-11-15    53,500       178,200        371,000        ...
Row 17:      2024-11-16    53,500       178,200        371,000        ...
...
```

- **Row 10:** Company names (column headers)
- **Column A (starting row 15):** Dates
- **Columns B onwards (starting row 15):** Daily stock prices

---

## Top 30 Companies Included

Major Korean stocks from KOSPI and KOSDAQ:

1. 삼성전자 (Samsung Electronics)
2. SK하이닉스 (SK Hynix)
3. LG에너지솔루션 (LG Energy Solution)
4. 삼성바이오로직스 (Samsung Biologics)
5. 현대차 (Hyundai Motor)
6. HD현대중공업 (HD Hyundai Heavy Industries)
7. 두산에너빌리티 (Doosan Enerbility)
8. KB금융 (KB Financial)
9. 한화에어로스페이스 (Hanwha Aerospace)
10. 기아 (Kia)
11. 셀트리온 (Celltrion)
12. NAVER
13. 한화오션 (Hanwha Ocean)
14. 신한지주 (Shinhan Financial Group)
15. SK스퀘어 (SK Square)
16. 삼성물산 (Samsung C&T)
17. 삼성생명 (Samsung Life)
18. HD한국조선해양 (HD Korea Shipbuilding)
19. 한국전력 (KEPCO)
20. HD현대일렉트릭 (HD Hyundai Electric)
21. 알테오젠 (Alteogen)
22. LG화학 (LG Chem)
23. 현대모비스 (Hyundai Mobis)
24. 카카오 (Kakao)
25. 하나금융지주 (Hana Financial Group)
26. POSCO홀딩스 (POSCO Holdings)
27. 삼성SDI (Samsung SDI)
28. 삼성중공업 (Samsung Heavy Industries)
29. 삼성화재 (Samsung Fire & Marine)
30. 고려아연 (Korea Zinc)

...and 2,590 more companies

---

## Sample Data - 삼성전자 (Samsung Electronics)

**Price Evolution:**
- **Earliest (Nov 14, 2024):** 49,900 KRW
- **Latest (Nov 14, 2025):** 97,200 KRW
- **Price Range:** 49,900 → 111,100 KRW
- **Return:** +94.8% over the year

---

## What You Can Do With This Data

### 1. Momentum Strategy (Recommended)
Test if stocks that went up recently keep going up:
```python
For each day:
  - Calculate past 20-day return for each stock
  - Buy top 10% performers (winners)
  - Sell bottom 10% performers (losers)
  - Measure next day returns

Result: Do winners beat losers?
```

### 2. Mean Reversion Strategy
Test if stocks that dropped recently bounce back:
```python
- Buy stocks that dropped most in past week
- Sell stocks that rose most in past week
- Check if they reverse
```

### 3. Volatility Strategy
Test if low-volatility stocks are safer:
```python
- Calculate 30-day volatility for each stock
- Compare high vs low volatility portfolio returns
```

### 4. Sector Analysis
Compare performance across industries:
```python
- Tech: Samsung, SK Hynix, NAVER
- Auto: Hyundai, Kia
- Finance: KB, Shinhan, Hana
- Bio: Samsung Biologics, Celltrion, Alteogen
```

---

## Data Format for Python Analysis

### How to Load the Data

```python
import pandas as pd
import numpy as np

# Read the Excel file
df = pd.read_excel('data/quant.xlsx',
                   sheet_name='Sheet1 (2)',
                   header=None)

# Extract company names (row 10, index 9)
companies = df.iloc[9, 1:].dropna().tolist()
print(f"Number of companies: {len(companies)}")

# Extract price data (rows 15+, columns 1+)
# Row 15 is index 14 in Python
dates = pd.to_datetime(df.iloc[14:, 0])
prices = df.iloc[14:, 1:len(companies)+1]

# Create proper DataFrame
stock_data = pd.DataFrame(
    prices.values,
    index=dates,
    columns=companies
)

print(stock_data.head())
```

### Result:
```
              삼성전자    SK하이닉스    LG에너지솔루션    삼성바이오로직스
2024-11-14    49900      173000       422000          957000
2024-11-15    53500      178200       371000          1041000
2024-11-16    53500      178200       371000          1041000
...
```

---

## Quick Analysis Example

### Calculate Daily Returns

```python
# Calculate daily returns (%)
returns = stock_data.pct_change() * 100

# Show Samsung Electronics returns
print(returns['삼성전자'].head(10))

# Average daily return for each stock
avg_returns = returns.mean()
print(avg_returns.sort_values(ascending=False).head(10))
```

### Calculate Momentum Signal

```python
# Calculate 20-day momentum (past 20 days return)
momentum_20d = stock_data.pct_change(20) * 100

# For each day, rank stocks by momentum
ranks = momentum_20d.rank(axis=1, pct=True)

# Identify winners (top 10%) and losers (bottom 10%)
winners = ranks >= 0.9
losers = ranks <= 0.1
```

---

## Data Characteristics

### Trading Days
- Total days: 366 (approximately 1 year)
- Includes weekdays and weekends (need to verify if weekends have data)
- Date format: Daily frequency

### Price Data
- Units: Korean Won (KRW)
- Format: Integer prices (no decimals)
- Example: 49900 = 49,900 KRW

### Missing Data (1.73%)
Reasons for missing values:
- Stock suspensions
- New IPOs (no historical data)
- Delisted stocks
- Non-trading days for specific stocks

**Handling:** Drop stocks with >20% missing data or forward-fill prices

---

## Next Steps for Your Project

### Step 1: Clean the Data
```python
# Load data
# Remove stocks with too much missing data
# Forward-fill remaining gaps
# Calculate returns
```

### Step 2: Implement Strategy
```python
# Choose: Momentum / Mean Reversion / Other
# Define ranking period (20 days? 60 days?)
# Define holding period (1 day? 5 days?)
# Form portfolios
```

### Step 3: Backtest
```python
# For each day:
#   - Rank stocks
#   - Form portfolios
#   - Calculate next-period return
# Aggregate results
```

### Step 4: Analyze Results
```python
# Calculate:
#   - Average returns
#   - Win rate
#   - Sharpe ratio
#   - Maximum drawdown
# Create visualizations
```

### Step 5: Present
```python
# Show:
#   - Strategy description
#   - Results table
#   - Cumulative return chart
#   - Statistical significance
```

---

## Advantages of This Dataset

✅ **Large universe:** 2,620 stocks (good for diversification)
✅ **High quality:** Only 1.73% missing
✅ **Daily frequency:** Can test short-term strategies
✅ **Korean market:** Different from US (may find unique patterns)
✅ **Recent data:** 2024-2025 (includes recent market conditions)
✅ **Well-known companies:** Can verify results make sense

---

## Comparison: FnGuide vs WRDS

| Feature | FnGuide (Your Data) | WRDS |
|---------|---------------------|------|
| Market | Korean stocks | US stocks |
| Companies | 2,620 | ~25,000 |
| Time Period | 1 year (2024-2025) | 60+ years (1960s-now) |
| Data Type | Daily prices | Monthly/Daily, fundamentals |
| Access | You already have it! | Need account approval |
| Cost | Free (from professor?) | Requires institutional access |

**Recommendation:** Start with FnGuide data since you already have it. You can complete your entire project without waiting for WRDS!

---

## Example Output You'll Create

```
MOMENTUM STRATEGY - KOREAN STOCKS (2024-2025)
==============================================

Strategy: Buy top 10%, Sell bottom 10% (20-day momentum)

Results:
  Winners Portfolio:    +1.2% per week
  Losers Portfolio:     +0.3% per week
  Long-Short Spread:    +0.9% per week

  Annual Return:        +46.8%
  Sharpe Ratio:         2.1
  Win Rate:             64%

Statistical Test:
  T-statistic:          4.5 (highly significant)
  P-value:              0.0001

Conclusion:
  ✅ Momentum effect exists in Korean market
  ✅ Top performers continue outperforming
  ✅ Strategy generates significant excess returns
```

---

## Questions This Data Can Answer

1. **Does momentum work in Korea?** (Your main project)
2. Do Samsung/SK Hynix moves predict tech sector?
3. Do small caps outperform large caps?
4. Is volatility rewarded in Korean market?
5. Do financial stocks move together?
6. Which sectors have strongest momentum?
7. How long does momentum persist? (1 day? 1 week? 1 month?)
8. Does momentum work better for large or small stocks?

---

## Ready to Start?

You have everything you need:
- ✅ Python installed
- ✅ Libraries installed
- ✅ Data file ready
- ✅ 2,620 stocks × 366 days = 958,920 data points

**Next:** Let me know if you want me to create the data loading and cleaning script!
