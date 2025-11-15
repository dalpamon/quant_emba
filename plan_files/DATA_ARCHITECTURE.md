# Data Architecture Document
## Factor Lab - Data Pipeline & Storage Design

**Version:** 1.0  
**Date:** October 25, 2024  
**Author:** System Architect  
**Status:** Design Phase  

---

## 1. Architecture Overview

### 1.1 High-Level Data Flow

```
┌─────────────────┐
│  Data Sources   │
│  (yfinance)     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Data Ingestion │
│  Layer          │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Local Cache    │
│  (SQLite)       │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Data Processing│
│  (Pandas)       │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Factor Engine  │
│  (NumPy/Scipy)  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Backtest Engine│
│  (Pure Python)  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Results Layer  │
│  (JSON/Dict)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Presentation   │
│  (Streamlit/Web)│
└─────────────────┘
```

---

## 2. Data Sources

### 2.1 Primary Source: yfinance (US Market)

**Endpoint:** Yahoo Finance API (via yfinance Python wrapper)

**Data Types:**
1. **OHLCV (Open, High, Low, Close, Volume)**
   - Frequency: Daily
   - Historical depth: 10+ years
   - Auto-adjusted for splits/dividends

2. **Fundamental Data**
   - Market Cap
   - P/E Ratio (Trailing)
   - P/B Ratio (Price-to-Book)
   - Profit Margins
   - Revenue, Earnings

3. **Corporate Actions**
   - Stock splits
   - Dividends
   - Spin-offs (if applicable)

**API Limits:**
- No official rate limit (but be respectful)
- Batch requests recommended (download multiple tickers at once)
- Typical response time: 5-15 seconds for 100 stocks

**Sample Request:**
```python
import yfinance as yf

# Single ticker
data = yf.download("AAPL", start="2020-01-01", end="2024-12-31")

# Multiple tickers (FASTER)
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(
    tickers,
    start="2020-01-01",
    end="2024-12-31",
    group_by='ticker',
    auto_adjust=True  # Critical: adjust for splits
)

# Fundamental data
ticker = yf.Ticker("AAPL")
info = ticker.info
market_cap = info['marketCap']
pe_ratio = info['trailingPE']
```

### 2.2 Future Sources (Project 2+)

**Korean Market:**
- FinanceDataReader (free)
- pykrx (official KRX data)
- FnGuide (institutional, paid)

**Alternative Data:**
- Quandl (economic indicators)
- Alpha Vantage (real-time quotes)
- FRED (Federal Reserve data)

---

## 3. Data Models & Schemas

### 3.1 Raw Price Data (OHLCV)

**Table:** `price_data`

```sql
CREATE TABLE price_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    adjusted_close DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Indexes for fast queries
CREATE INDEX idx_ticker_date ON price_data(ticker, date);
CREATE INDEX idx_date ON price_data(date);
```

**Sample Data:**
```
| ticker | date       | open   | high   | low    | close  | volume    | adjusted_close |
|--------|------------|--------|--------|--------|--------|-----------|----------------|
| AAPL   | 2024-01-02 | 185.50 | 187.20 | 184.90 | 186.75 | 45678900  | 186.75         |
| AAPL   | 2024-01-03 | 186.80 | 188.50 | 186.20 | 187.90 | 52341000  | 187.90         |
```

---

### 3.2 Fundamental Data

**Table:** `fundamentals`

```sql
CREATE TABLE fundamentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    market_cap BIGINT,
    pe_ratio DECIMAL(10, 2),
    pb_ratio DECIMAL(10, 2),
    profit_margin DECIMAL(8, 4),
    roe DECIMAL(8, 4),
    revenue BIGINT,
    net_income BIGINT,
    total_assets BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_fund_ticker_date ON fundamentals(ticker, date);
```

**Sample Data:**
```
| ticker | date       | market_cap   | pe_ratio | pb_ratio | profit_margin |
|--------|------------|--------------|----------|----------|---------------|
| AAPL   | 2024-01-31 | 2900000000000| 28.50    | 45.30    | 0.2580        |
```

---

### 3.3 Calculated Factors

**Table:** `factors`

```sql
CREATE TABLE factors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    momentum_12m DECIMAL(10, 4),
    value_bm DECIMAL(10, 4),
    size_log_mcap DECIMAL(10, 4),
    quality_roa DECIMAL(10, 4),
    volatility_60d DECIMAL(10, 4),
    composite_score DECIMAL(10, 4),  -- Weighted combination
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_factor_date ON factors(date);
CREATE INDEX idx_composite_score ON factors(date, composite_score);
```

**Sample Data:**
```
| ticker | date       | momentum_12m | value_bm | size_log_mcap | quality_roa | volatility_60d | composite_score |
|--------|------------|--------------|----------|---------------|-------------|----------------|-----------------|
| AAPL   | 2024-01-31 | 0.2340       | 0.0221   | 28.95         | 0.2680      | 0.0145         | 0.8234          |
| MSFT   | 2024-01-31 | 0.1890       | 0.0189   | 29.12         | 0.3120      | 0.0132         | 0.7654          |
```

---

### 3.4 Portfolio Holdings

**Table:** `portfolio_holdings`

```sql
CREATE TABLE portfolio_holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id VARCHAR(50),
    rebalance_date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    position_type VARCHAR(10),  -- 'long' or 'short'
    weight DECIMAL(8, 6),
    shares INTEGER,
    entry_price DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_id, rebalance_date, ticker)
);

CREATE INDEX idx_strategy_date ON portfolio_holdings(strategy_id, rebalance_date);
```

**Sample Data:**
```
| strategy_id     | rebalance_date | ticker | position_type | weight   | shares | entry_price |
|-----------------|----------------|--------|---------------|----------|--------|-------------|
| momentum_value  | 2024-01-31     | AAPL   | long          | 0.1000   | 50     | 186.75      |
| momentum_value  | 2024-01-31     | INTC   | short         | -0.0500  | -100   | 42.30       |
```

---

### 3.5 Backtest Results

**Table:** `backtest_results`

```sql
CREATE TABLE backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id VARCHAR(50) UNIQUE,
    strategy_name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    initial_capital DECIMAL(15, 2),
    final_value DECIMAL(15, 2),
    total_return DECIMAL(10, 4),
    cagr DECIMAL(10, 4),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    volatility DECIMAL(8, 4),
    num_trades INTEGER,
    win_rate DECIMAL(6, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_strategy_id ON backtest_results(strategy_id);
```

**Sample Data:**
```
| strategy_id    | strategy_name        | total_return | cagr   | sharpe_ratio | max_drawdown |
|----------------|----------------------|--------------|--------|--------------|--------------|
| momentum_value | Momentum + Value 60% | 0.4520       | 0.1890 | 1.45         | -0.1890      |
```

---

### 3.6 Daily Equity Curve

**Table:** `equity_curve`

```sql
CREATE TABLE equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id VARCHAR(50),
    date DATE NOT NULL,
    portfolio_value DECIMAL(15, 2),
    daily_return DECIMAL(10, 6),
    cumulative_return DECIMAL(10, 4),
    drawdown DECIMAL(8, 4),
    UNIQUE(strategy_id, date)
);

CREATE INDEX idx_equity_strategy_date ON equity_curve(strategy_id, date);
```

**Sample Data:**
```
| strategy_id    | date       | portfolio_value | daily_return | cumulative_return | drawdown |
|----------------|------------|-----------------|--------------|-------------------|----------|
| momentum_value | 2024-01-02 | 100000.00       | 0.0000       | 0.0000            | 0.0000   |
| momentum_value | 2024-01-03 | 101250.00       | 0.0125       | 0.0125            | 0.0000   |
| momentum_value | 2024-01-04 | 100875.00       | -0.0037      | 0.0088            | -0.0037  |
```

---

## 4. ETL Pipeline

### 4.1 Data Ingestion Workflow

```python
# data_pipeline.py

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

class DataPipeline:
    def __init__(self, db_path='factorlab.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_db()
    
    def _initialize_db(self):
        """Create tables if they don't exist"""
        # Execute CREATE TABLE statements from section 3
        pass
    
    def fetch_and_cache(self, tickers, start_date, end_date, force_refresh=False):
        """
        Fetch data from yfinance and cache locally
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached
        
        Returns:
            pd.DataFrame: Multi-index dataframe (ticker, date)
        """
        # Check cache first
        if not force_refresh:
            cached_data = self._load_from_cache(tickers, start_date, end_date)
            if cached_data is not None:
                print("✅ Loaded from cache")
                return cached_data
        
        # Fetch from yfinance
        print(f"📥 Downloading {len(tickers)} tickers from {start_date} to {end_date}...")
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            progress=True
        )
        
        # Save to cache
        self._save_to_cache(data, tickers)
        
        return data
    
    def _load_from_cache(self, tickers, start_date, end_date):
        """Load data from SQLite cache"""
        query = f"""
        SELECT * FROM price_data
        WHERE ticker IN ({','.join(['?']*len(tickers))})
        AND date BETWEEN ? AND ?
        ORDER BY ticker, date
        """
        try:
            df = pd.read_sql_query(
                query,
                self.conn,
                params=tickers + [start_date, end_date],
                parse_dates=['date']
            )
            
            if df.empty:
                return None
            
            # Reshape to multi-index format
            df = df.pivot(index='date', columns='ticker')
            return df
        except:
            return None
    
    def _save_to_cache(self, data, tickers):
        """Save downloaded data to SQLite"""
        # Flatten multi-index and save to DB
        for ticker in tickers:
            ticker_data = data[ticker].reset_index()
            ticker_data['ticker'] = ticker
            ticker_data.to_sql(
                'price_data',
                self.conn,
                if_exists='append',
                index=False
            )
        
        print("✅ Saved to cache")
    
    def fetch_fundamentals(self, tickers):
        """Fetch fundamental data for tickers"""
        fundamentals = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                fundamentals.append({
                    'ticker': ticker,
                    'date': datetime.now().date(),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'profit_margin': info.get('profitMargins'),
                    'roe': info.get('returnOnEquity'),
                })
            except Exception as e:
                print(f"❌ Failed to fetch {ticker}: {e}")
        
        df = pd.DataFrame(fundamentals)
        
        # Save to DB
        df.to_sql('fundamentals', self.conn, if_exists='append', index=False)
        
        return df
```

### 4.2 Data Refresh Strategy

**Cache Invalidation Rules:**
1. **Intraday:** Use cache (no refresh needed)
2. **After market close (4:30 PM ET):** Fetch latest day
3. **Weekly:** Full refresh of fundamentals
4. **On-demand:** User can force refresh via UI

**Implementation:**
```python
def should_refresh(last_cached_date):
    """Check if cache needs refresh"""
    now = datetime.now()
    market_close = now.replace(hour=16, minute=30)
    
    # If after market close and cache is from yesterday or earlier
    if now > market_close and last_cached_date < now.date():
        return True
    
    return False
```

---

## 5. Data Processing Pipeline

### 5.1 Factor Calculation Flow

```python
# factor_engine.py

class FactorEngine:
    """Calculate quantitative factors from price data"""
    
    @staticmethod
    def calculate_all_factors(price_data, fundamental_data):
        """
        Calculate all factors in one pass (efficient)
        
        Args:
            price_data: pd.DataFrame with OHLCV
            fundamental_data: pd.DataFrame with fundamentals
        
        Returns:
            pd.DataFrame: Factor scores per ticker per date
        """
        factors = pd.DataFrame()
        
        # 1. Momentum (price-based, fast)
        factors['momentum_12m'] = FactorEngine._momentum(price_data)
        
        # 2. Volatility (price-based, fast)
        factors['volatility_60d'] = FactorEngine._volatility(price_data)
        
        # 3. Value (fundamental, updated monthly)
        factors['value_bm'] = FactorEngine._value(fundamental_data)
        
        # 4. Size (fundamental, updated monthly)
        factors['size_log_mcap'] = FactorEngine._size(fundamental_data)
        
        # 5. Quality (fundamental, updated monthly)
        factors['quality_roa'] = FactorEngine._quality(fundamental_data)
        
        # Normalize each factor (z-score)
        for col in factors.columns:
            factors[col] = (factors[col] - factors[col].mean()) / factors[col].std()
        
        return factors
    
    @staticmethod
    def _momentum(prices, lookback=252, skip_last=21):
        """12-month momentum, skip last month"""
        returns = prices.pct_change(lookback).shift(skip_last)
        return returns
    
    @staticmethod
    def _volatility(prices, window=60):
        """60-day rolling volatility"""
        returns = prices.pct_change()
        vol = returns.rolling(window).std()
        return -vol  # Negative because low vol is good
    
    @staticmethod
    def _value(fundamentals):
        """Book-to-Market ratio"""
        return 1 / fundamentals['pb_ratio']
    
    @staticmethod
    def _size(fundamentals):
        """Log market cap (smaller is better)"""
        return -np.log(fundamentals['market_cap'])
    
    @staticmethod
    def _quality(fundamentals):
        """Return on Assets"""
        roa = fundamentals['net_income'] / fundamentals['total_assets']
        return roa
```

### 5.2 Data Quality Checks

```python
class DataQualityChecker:
    """Ensure data integrity before backtesting"""
    
    @staticmethod
    def check_missing_data(df, threshold=0.05):
        """Alert if >5% data is missing"""
        missing_pct = df.isnull().sum() / len(df)
        
        if (missing_pct > threshold).any():
            print(f"⚠️ Warning: {missing_pct[missing_pct > threshold]}")
            return False
        return True
    
    @staticmethod
    def check_outliers(df, z_threshold=5):
        """Detect extreme outliers (likely data errors)"""
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = (z_scores > z_threshold).sum()
        
        if outliers.sum() > 0:
            print(f"⚠️ Found {outliers.sum()} outliers")
            return False
        return True
    
    @staticmethod
    def check_splits(df):
        """Detect unhandled stock splits"""
        daily_returns = df.pct_change()
        
        # If return > 50% in one day, likely a split issue
        extreme_returns = (daily_returns.abs() > 0.5).sum()
        
        if extreme_returns > 0:
            print(f"⚠️ Possible unhandled splits: {extreme_returns}")
            return False
        return True
```

---

## 6. Storage Architecture

### 6.1 File System Structure

```
factor-lab/
├── data/
│   ├── cache/
│   │   ├── factorlab.db          # SQLite database
│   │   ├── price_data_backup.csv # Daily backup
│   │   └── last_updated.txt      # Cache timestamp
│   │
│   ├── exports/
│   │   ├── backtest_results/
│   │   │   ├── strategy_001.json
│   │   │   └── strategy_002.json
│   │   └── equity_curves/
│   │       ├── strategy_001.csv
│   │       └── strategy_002.csv
│   │
│   └── raw/
│       └── manual_uploads/       # For CSV imports
│
├── config/
│   ├── tickers.json              # Predefined universes
│   └── settings.json             # App configuration
│
└── logs/
    ├── data_pipeline.log
    ├── backtest.log
    └── errors.log
```

### 6.2 Database Choice: SQLite

**Why SQLite?**
- ✅ Zero configuration (serverless)
- ✅ Fast for read-heavy workloads
- ✅ Single file (easy to backup)
- ✅ Sufficient for <1M rows
- ✅ Built into Python

**Limitations:**
- ⚠️ No concurrent writes (but we don't need it)
- ⚠️ Scales to ~100GB (more than enough)

**Migration Path (Project 3):**
- If users > 1000 or data > 10GB → PostgreSQL
- If need real-time collaboration → Firebase/Supabase

---

## 7. Performance Optimization

### 7.1 Caching Strategy

**Three-Tier Cache:**

1. **In-Memory (Python dicts)**
   - Current session data
   - Lifetime: Until page refresh
   - Size: ~100MB

2. **SQLite (Disk cache)**
   - Historical data (months)
   - Lifetime: 7 days
   - Size: ~500MB

3. **File System (Long-term backup)**
   - Exported results
   - Lifetime: Permanent
   - Size: Unlimited

### 7.2 Query Optimization

**Indexes:**
```sql
-- Speed up common queries
CREATE INDEX idx_price_ticker_date ON price_data(ticker, date);
CREATE INDEX idx_factor_composite ON factors(date, composite_score DESC);
```

**Batch Operations:**
```python
# BAD: One query per ticker
for ticker in tickers:
    data = query_single_ticker(ticker)

# GOOD: Single query for all tickers
data = query_multiple_tickers(tickers)
```

### 7.3 Data Compression

**For large datasets:**
```python
# Compress price data (save 70% space)
df.to_parquet('prices.parquet', compression='gzip')

# Instead of:
df.to_csv('prices.csv')  # 3x larger
```

---

## 8. Data Security & Privacy

### 8.1 Project 1 (MVP)
- No sensitive data (public stock prices)
- Local storage only (user's device)
- No user authentication

### 8.2 Project 3 (Production)
- **User Data:** Encrypted at rest (SQLCipher)
- **API Keys:** Environment variables only
- **Backups:** Automated daily (encrypted S3)
- **Compliance:** GDPR-ready (data export/delete)

---

## 9. Monitoring & Logging

### 9.1 Data Pipeline Logs

```python
import logging

logging.basicConfig(
    filename='logs/data_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('DataPipeline')

# Log every data fetch
logger.info(f"Fetching {len(tickers)} tickers from {start} to {end}")
logger.warning(f"Cache miss for {ticker}")
logger.error(f"Failed to download {ticker}: {error}")
```

### 9.2 Data Quality Metrics

Track daily:
- Data completeness (% of expected records)
- Download success rate
- Average query time
- Cache hit rate

---

## 10. API Specifications

### 10.1 Internal Data API

```python
# data_api.py

class DataAPI:
    """Unified interface for data access"""
    
    def get_prices(self, tickers, start_date, end_date):
        """
        Get OHLCV data
        
        Returns:
            pd.DataFrame: Multi-index (date, ticker)
        """
        pass
    
    def get_fundamentals(self, tickers, date=None):
        """
        Get fundamental data as of date
        
        Returns:
            pd.DataFrame: One row per ticker
        """
        pass
    
    def get_factors(self, tickers, start_date, end_date):
        """
        Get pre-calculated factors
        
        Returns:
            pd.DataFrame: Factor scores
        """
        pass
    
    def get_universe(self, universe_name):
        """
        Get predefined stock universe
        
        Args:
            universe_name: 'sp500', 'nasdaq100', 'tech'
        
        Returns:
            List[str]: Ticker symbols
        """
        pass
```

---

## 11. Data Governance

### 11.1 Data Retention Policy

| Data Type | Retention Period | Reason |
|-----------|------------------|---------|
| Price data | 10 years | Historical analysis |
| Fundamentals | 5 years | Long-term trends |
| Factors | 5 years | Strategy validation |
| Backtest results | 1 year | Performance tracking |
| Logs | 30 days | Debugging |

### 11.2 Data Sources Attribution

Always credit data sources:
- "Data provided by Yahoo Finance"
- "Powered by yfinance library"
- (Later) "Korean market data from KRX"

---

## 12. Appendix: Sample Queries

### 12.1 Get Top Momentum Stocks

```sql
SELECT 
    ticker,
    date,
    momentum_12m,
    RANK() OVER (PARTITION BY date ORDER BY momentum_12m DESC) as rank
FROM factors
WHERE date = '2024-01-31'
AND momentum_12m IS NOT NULL
ORDER BY rank
LIMIT 20;
```

### 12.2 Get Portfolio Returns Over Time

```sql
SELECT 
    ec.date,
    ec.portfolio_value,
    ec.daily_return,
    ec.cumulative_return
FROM equity_curve ec
WHERE ec.strategy_id = 'momentum_value'
ORDER BY ec.date;
```

### 12.3 Get Factor Correlation

```sql
SELECT 
    CORR(momentum_12m, value_bm) as momentum_value_corr,
    CORR(momentum_12m, volatility_60d) as momentum_vol_corr,
    CORR(value_bm, quality_roa) as value_quality_corr
FROM factors
WHERE date BETWEEN '2023-01-01' AND '2024-12-31';
```

---

**Document Status:** ✅ Final  
**Next Steps:** Implement data pipeline (Day 1-2)  
**Owner:** You (Backend Developer)  
**Related Docs:** `PRD.md`, `IMPLEMENTATION_PLAN.md`
