# Technology Stack Document
## Factor Lab - Technical Architecture & Tool Selection

**Version:** 1.0  
**Date:** October 25, 2024  
**Architect:** Technical Lead  
**Decision Status:** Final for Project 1  

---

## 1. Executive Summary

### 1.1 Stack Overview

**For Project 1 (Week 1 MVP):**
```
Frontend:  Streamlit (Python-based)
Backend:   Python 3.10+
Data:      yfinance + pandas
Storage:   SQLite
Charts:    Plotly
Deploy:    Streamlit Cloud (FREE)
```

**Why This Stack:**
- ✅ **Speed:** Build in 1 week (proven possible)
- ✅ **Learning:** Stay in Python ecosystem
- ✅ **Cost:** $0 (completely free)
- ✅ **Mobile:** Responsive enough for Week 1
- ✅ **Migration Path:** Easy to upgrade later

---

## 2. Frontend Framework Decision

### 2.1 Options Considered

| Framework | Build Time | Mobile Support | Python Integration | Learning Curve |
|-----------|-----------|----------------|-------------------|----------------|
| **Streamlit** | 1 week | Good (7/10) | Native | Low |
| Next.js + React | 3 weeks | Excellent (10/10) | API only | High |
| Flask + Jinja | 2 weeks | Good (8/10) | Native | Medium |
| Dash (Plotly) | 1.5 weeks | Good (7/10) | Native | Medium |

### 2.2 Decision: Streamlit for Project 1

**Why Streamlit:**

**1. Development Speed (Critical for Week 1)**
```python
# Entire app structure in <100 lines
import streamlit as st

st.title("Factor Lab")
factors = st.slider("Momentum", 0, 100, 40)
if st.button("Run Backtest"):
    results = run_backtest(factors)
    st.plotly_chart(results.equity_curve)
```

**2. Built-in Components**
- Sliders, dropdowns, date pickers (no need to build)
- Automatic layout (mobile-responsive by default)
- Built-in caching (`@st.cache_data`)

**3. Python-Native**
- No JavaScript required
- Direct access to pandas/numpy
- Call data pipeline functions directly

**4. Free Hosting**
- Streamlit Cloud (1GB RAM, unlimited public apps)
- Deploy with Git push
- Custom domain support

**Limitations (Accept for Week 1):**
- ❌ Less control over mobile UX
- ❌ Slower than React (but <3s is fine)
- ❌ Limited customization (but adequate)

**Migration Path (Project 3):**
- Backend stays Python (FastAPI)
- Frontend → Next.js + React
- Reuse 80% of business logic

---

### 2.3 Streamlit Configuration

**Installation:**
```bash
pip install streamlit==1.28.0
pip install plotly==5.17.0
pip install pandas==2.1.0
```

**Project Structure:**
```
factor-lab/
├── app.py                  # Main Streamlit app
├── pages/
│   ├── 1_Strategy_Builder.py
│   ├── 2_Results.py
│   └── 3_Factor_Explorer.py
├── components/
│   ├── factor_slider.py
│   ├── metric_card.py
│   └── equity_chart.py
├── core/                   # Business logic
│   ├── data_pipeline.py
│   ├── factor_engine.py
│   ├── backtester.py
│   └── analytics.py
├── data/
│   └── cache.db
├── requirements.txt
└── README.md
```

**Streamlit Config (`/.streamlit/config.toml`):**
```toml
[theme]
primaryColor = "#00D9FF"
backgroundColor = "#F8F9FA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#212121"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true
```

---

## 3. Backend Technology

### 3.1 Language: Python 3.10+

**Why Python:**
- ✅ Your existing expertise
- ✅ Best data science ecosystem (pandas, numpy, scipy)
- ✅ Quant industry standard
- ✅ Easy prototyping

**Python Version:**
```bash
# Use 3.10 or 3.11 (not 3.12 yet - library compatibility)
python --version
# Python 3.10.12
```

**Virtual Environment:**
```bash
# Create environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 3.2 Core Libraries

**Data Manipulation:**
```python
pandas==2.1.0          # DataFrames, time-series
numpy==1.24.0          # Numerical computing
scipy==1.11.0          # Statistical functions
```

**Data Acquisition:**
```python
yfinance==0.2.28       # Yahoo Finance data
requests==2.31.0       # HTTP client (for APIs)
```

**Data Storage:**
```python
# SQLite (built-in to Python)
import sqlite3
```

**Visualization:**
```python
plotly==5.17.0         # Interactive charts
matplotlib==3.8.0      # Backup for static charts
```

**Utilities:**
```python
python-dateutil==2.8.2 # Date manipulation
pytz==2023.3           # Timezone handling
tqdm==4.66.1           # Progress bars
```

**Testing:**
```python
pytest==7.4.2          # Unit testing
pytest-cov==4.1.0      # Code coverage
```

---

## 4. Data Layer

### 4.1 Data Source: yfinance

**Why yfinance:**
- ✅ FREE unlimited requests
- ✅ 10+ years historical data
- ✅ Auto-adjusted (splits, dividends)
- ✅ Fundamental data included
- ✅ Well-maintained (active community)

**API Example:**
```python
import yfinance as yf

# Efficient: Download multiple tickers at once
data = yf.download(
    ["AAPL", "MSFT", "GOOGL"],
    start="2020-01-01",
    end="2024-12-31",
    group_by='ticker',
    auto_adjust=True,  # Critical: adjust for splits
    threads=True        # Parallel downloads
)

# Fundamental data
ticker = yf.Ticker("AAPL")
info = ticker.info  # Dict with 100+ fields
```

**Rate Limiting:**
- No official limit
- Best practice: Batch requests, cache aggressively
- If hit rate limit: Add 1-second delay between requests

**Alternatives (Project 2+):**
```python
# Korean market
import FinanceDataReader as fdr
from pykrx import stock

# Premium data (if budget allows)
# import quandl
# import alpha_vantage
```

---

### 4.2 Database: SQLite

**Why SQLite:**
- ✅ Zero configuration (file-based)
- ✅ Fast for read-heavy workloads
- ✅ Built into Python (no install needed)
- ✅ Perfect for <1GB data
- ✅ Easy backup (just copy .db file)

**Schema Management:**
```python
# database.py
import sqlite3

class Database:
    def __init__(self, db_path='data/factorlab.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create tables if they don't exist"""
        
        # Price data table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adjusted_close REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        ''')
        
        # Create indexes for performance
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ticker_date 
            ON price_data(ticker, date)
        ''')
        
        self.conn.commit()
```

**ORM Alternative (Optional):**
```python
# If you want object-relational mapping
from sqlalchemy import create_engine, Column, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class PriceData(Base):
    __tablename__ = 'price_data'
    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    close = Column(Float)
    # ...
```

**Migration Path:**
```
Week 1-2:  SQLite (local file)
           ↓
Project 2: Still SQLite (sufficient)
           ↓
Project 3: PostgreSQL (if users > 1000)
           or Firebase (if need real-time sync)
```

---

### 4.3 Caching Strategy

**Streamlit Built-in Cache:**
```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data(tickers, start_date, end_date):
    """Downloads are expensive - cache results"""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

@st.cache_data
def calculate_factors(price_data):
    """Factor calculations are deterministic - cache forever"""
    factors = FactorEngine.calculate_all(price_data)
    return factors
```

**File-based Cache (for persistence):**
```python
import pickle
import hashlib
from pathlib import Path

def get_cache_key(tickers, start, end):
    """Generate unique cache key"""
    key = f"{','.join(sorted(tickers))}_{start}_{end}"
    return hashlib.md5(key.encode()).hexdigest()

def cache_data(data, cache_key):
    """Save to disk"""
    cache_dir = Path("data/cache")
    cache_dir.mkdir(exist_ok=True)
    
    with open(cache_dir / f"{cache_key}.pkl", 'wb') as f:
        pickle.dump(data, f)

def load_cached_data(cache_key):
    """Load from disk if exists"""
    cache_file = Path(f"data/cache/{cache_key}.pkl")
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None
```

---

## 5. Computational Layer

### 5.1 Factor Calculation Engine

**Library Choice: Pure NumPy/Pandas**

**Why NOT use a library:**
- ❌ Qlib: Too complex, overkill
- ❌ Zipline: Deprecated, poor maintenance
- ❌ Backtrader: Steep learning curve

**Why build from scratch:**
- ✅ Learn the actual math
- ✅ Full control & customization
- ✅ Easier debugging
- ✅ Portable to other projects

**Performance Optimization:**
```python
import numpy as np
import pandas as pd
from numba import jit  # Optional: JIT compilation

# Vectorized operations (FAST)
def calculate_momentum_vectorized(prices):
    """Uses numpy - 100x faster than loops"""
    returns = (prices.iloc[-1] / prices.iloc[-252] - 1)
    return returns

# If still slow, use Numba
@jit(nopython=True)
def calculate_rolling_std(returns):
    """JIT-compiled - 10x faster than pandas"""
    # ... numpy operations only
    pass
```

---

### 5.2 Backtesting Engine

**Custom Implementation**

**Why not use backtesting libraries:**
- Zipline: Deprecated
- Backtrader: Overcomplicated API
- bt: Good, but limited factor support

**Simple, Correct Backtester:**
```python
class Backtester:
    """Vectorized backtesting engine"""
    
    def run(self, prices, positions, rebalance_dates):
        """
        Vectorized backtest (fast)
        
        Args:
            prices: DataFrame (date x tickers)
            positions: Dict {date: {'long': [...], 'short': [...]}}
            rebalance_dates: List of rebalancing dates
        
        Returns:
            DataFrame: Daily portfolio value
        """
        # Calculate returns for all stocks
        returns = prices.pct_change()
        
        # Initialize portfolio
        portfolio_value = [100000]  # Starting capital
        
        current_long = []
        current_short = []
        
        for i, date in enumerate(prices.index[1:], 1):
            # Rebalance if needed
            if date in rebalance_dates:
                current_long = positions[date]['long']
                current_short = positions[date]['short']
            
            # Calculate daily return
            if current_long or current_short:
                long_ret = returns.loc[date, current_long].mean() if current_long else 0
                short_ret = -returns.loc[date, current_short].mean() if current_short else 0
                portfolio_ret = (long_ret + short_ret) / 2
            else:
                portfolio_ret = 0
            
            # Update portfolio value
            portfolio_value.append(portfolio_value[-1] * (1 + portfolio_ret))
        
        return pd.Series(portfolio_value, index=prices.index)
```

**Transaction Cost Model:**
```python
def apply_transaction_costs(returns, turnover, cost_bps=10):
    """
    Reduce returns by transaction costs
    
    Args:
        returns: Daily returns
        turnover: % of portfolio traded
        cost_bps: Cost in basis points (10 bps = 0.1%)
    
    Returns:
        Adjusted returns
    """
    cost = turnover * (cost_bps / 10000)
    return returns - cost
```

---

## 6. Visualization Layer

### 6.1 Chart Library: Plotly

**Why Plotly:**
- ✅ Interactive by default (zoom, pan, hover)
- ✅ Mobile-friendly
- ✅ Streamlit integration (`st.plotly_chart()`)
- ✅ Beautiful out-of-box
- ✅ Supports 40+ chart types

**Example Chart:**
```python
import plotly.graph_objects as go

def create_equity_curve_chart(equity_curve, benchmark=None):
    """Create interactive equity curve"""
    
    fig = go.Figure()
    
    # Main equity curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Strategy',
        line=dict(color='#00D9FF', width=2),
        hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'
    ))
    
    # Optional benchmark
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#BDBDBD', width=1, dash='dash')
        ))
    
    # Layout
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig
```

**Mobile Optimization:**
```python
# Responsive config
config = {
    'displayModeBar': False,  # Hide toolbar on mobile
    'responsive': True,
    'doubleClick': 'reset'
}

st.plotly_chart(fig, use_container_width=True, config=config)
```

**Chart Types Used:**
1. **Equity Curve:** Line chart (go.Scatter)
2. **Drawdown:** Area chart (go.Scatter fill)
3. **Returns Distribution:** Histogram (go.Histogram)
4. **Factor Attribution:** Bar chart (go.Bar)
5. **Correlation Matrix:** Heatmap (go.Heatmap)

---

### 6.2 Alternative: Matplotlib (Backup)

**Use matplotlib for:**
- Static exports (PNG for presentations)
- When Plotly is too heavy

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_equity_curve_static(equity_curve):
    """Static equity curve for export"""
    fig, ax = plt.subplots()
    ax.plot(equity_curve.index, equity_curve.values, 
            color='#00D9FF', linewidth=2)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    plt.tight_layout()
    return fig
```

---

## 7. Deployment & Hosting

### 7.1 Streamlit Cloud (FREE)

**Why Streamlit Cloud:**
- ✅ **FREE** for public apps
- ✅ 1GB RAM (sufficient)
- ✅ Auto-deploy on Git push
- ✅ HTTPS + custom domain
- ✅ No DevOps required

**Setup Steps:**
```bash
# 1. Push code to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/factor-lab.git
git push -u origin main

# 2. Go to share.streamlit.io
# 3. Connect GitHub repo
# 4. Click "Deploy"
# Done! App live at: https://factor-lab.streamlit.app
```

**Secrets Management:**
```toml
# .streamlit/secrets.toml (not committed)
[api_keys]
alphavantage = "YOUR_API_KEY"  # If using paid APIs later

# Access in code:
import streamlit as st
api_key = st.secrets["api_keys"]["alphavantage"]
```

**Resource Limits:**
- 1GB RAM (enough for 500 stocks x 5 years)
- No persistent disk (use GitHub or DB)
- 1 CPU core
- Always-on (sleeps after 7 days inactivity)

---

### 7.2 Alternative: Docker + Cloud Run (Project 3)

**For more control:**
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

**Deploy to Google Cloud Run:**
```bash
gcloud run deploy factor-lab \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**Cost:** ~$5/month (with free tier)

---

## 8. Development Tools

### 8.1 IDE Setup

**VS Code (Recommended)**

**Extensions:**
```
Python (Microsoft)
Pylance (Microsoft)
Jupyter (for notebooks)
Tailwind CSS IntelliSense
GitLens
Prettier
```

**Settings (`.vscode/settings.json`):**
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

**Alternative: PyCharm**
- Better for large projects
- Built-in database tools
- Professional debugger

---

### 8.2 Code Quality Tools

**Linting & Formatting:**
```bash
pip install black pylint mypy

# Format code
black .

# Lint
pylint app.py core/*.py

# Type checking
mypy core/
```

**Pre-commit Hooks:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.10
  
  - repo: https://github.com/pycqa/pylint
    rev: v3.0.0
    hooks:
      - id: pylint
```

---

### 8.3 Testing Framework

**pytest for Unit Tests:**
```python
# tests/test_factors.py
import pytest
import pandas as pd
import numpy as np
from core.factor_engine import FactorEngine

def test_momentum_calculation():
    """Test momentum factor calculation"""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252)
    prices = pd.Series(np.linspace(100, 120, 252), index=dates)
    
    # Calculate momentum
    momentum = FactorEngine.momentum(prices, lookback=252)
    
    # Assert
    expected = 0.20  # 20% gain
    assert abs(momentum.iloc[-1] - expected) < 0.01

def test_value_factor():
    """Test value factor calculation"""
    fundamentals = pd.DataFrame({
        'priceToBook': [2.0, 4.0, 1.5]
    }, index=['AAPL', 'MSFT', 'GOOGL'])
    
    value = FactorEngine.value(fundamentals)
    
    # Lower P/B = higher value score
    assert value['GOOGL'] > value['AAPL'] > value['MSFT']
```

**Run Tests:**
```bash
# Run all tests
pytest

# With coverage
pytest --cov=core --cov-report=html

# Open coverage report
open htmlcov/index.html
```

---

## 9. Version Control

### 9.1 Git Strategy

**Branch Structure:**
```
main             (production-ready)
  ├── dev        (active development)
  ├── feature/data-pipeline
  ├── feature/ui-builder
  └── feature/backtest-engine
```

**Commit Convention:**
```
feat: Add momentum factor calculation
fix: Handle missing data in yfinance download
docs: Update README with installation steps
test: Add unit tests for factor engine
refactor: Optimize backtest loop
```

---

### 9.2 .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data & Secrets
data/cache/
*.db
*.db-journal
.streamlit/secrets.toml

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
logs/
*.log
```

---

## 10. Performance Benchmarks

### 10.1 Target Metrics

| Operation | Target Time | Acceptable Max |
|-----------|-------------|----------------|
| Data download (100 stocks, 5 yrs) | 15s | 30s |
| Factor calculation (500 stocks) | 2s | 5s |
| Backtest (500 stocks, 5 yrs) | 3s | 10s |
| Chart rendering | 0.5s | 2s |
| Page load (first visit) | 2s | 5s |
| Page load (cached) | 0.5s | 2s |

### 10.2 Optimization Techniques

**1. Vectorization (NumPy)**
```python
# BAD: Python loops (100x slower)
momentum = []
for ticker in tickers:
    ret = (prices[ticker][-1] / prices[ticker][-252]) - 1
    momentum.append(ret)

# GOOD: Vectorized (NumPy)
momentum = (prices.iloc[-1] / prices.iloc[-252]) - 1
```

**2. Caching**
```python
# Cache expensive operations
@st.cache_data(ttl=3600)
def expensive_calculation(data):
    # ...
    return result
```

**3. Parallel Processing**
```python
from concurrent.futures import ThreadPoolExecutor

def download_ticker(ticker):
    return yf.download(ticker, ...)

with ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(download_ticker, tickers)
```

**4. Database Indexing**
```sql
CREATE INDEX idx_ticker_date ON price_data(ticker, date);
```

---

## 11. Security Considerations

### 11.1 Project 1 (No User Auth)

**Data Security:**
- ✅ No sensitive data (public stock prices)
- ✅ No user accounts
- ✅ No payment processing

**API Keys:**
```python
# Store in secrets (not in code)
import streamlit as st
api_key = st.secrets["api"]["key"]

# Never commit secrets
# Add to .gitignore: .streamlit/secrets.toml
```

---

### 11.2 Project 3 (Production)

**User Authentication:**
```bash
pip install streamlit-authenticator
# Or use: Auth0, Firebase Auth, Clerk
```

**Database Security:**
```python
# Encrypt SQLite database
pip install sqlcipher3

# Use prepared statements (prevent SQL injection)
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

**HTTPS:**
- Streamlit Cloud: Automatic
- Custom domain: Let's Encrypt (free)

---

## 12. Monitoring & Logging

### 12.1 Application Logs

```python
import logging

# Setup logging
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log important events
logger.info("User ran backtest: momentum=40%, value=30%")
logger.warning("Data download slow: 45 seconds")
logger.error("Backtest failed: division by zero")
```

### 12.2 Error Tracking (Project 3)

```bash
pip install sentry-sdk

# Automatic error reporting
import sentry_sdk
sentry_sdk.init(dsn="YOUR_DSN")
```

---

## 13. Migration Path

### 13.1 Evolution Timeline

**Project 1 (Week 1):**
```
Streamlit + SQLite + yfinance
→ Quick MVP, learning focus
```

**Project 2 (Nov-Dec):**
```
Same stack + Korean data sources
→ Course project ready
```

**Project 3 (Jan 2026+):**
```
Option A: Keep Streamlit (if sufficient)
  + Add authentication
  + Add payment (Stripe)
  + Better hosting (AWS/GCP)

Option B: Migrate to Next.js
  Frontend: Next.js + React + Tailwind
  Backend: FastAPI (Python)
  Database: PostgreSQL
  Hosting: Vercel + Railway
```

### 13.2 When to Migrate

**Stay with Streamlit if:**
- User count < 1,000
- Users are technical (tolerate limitations)
- Budget is tight
- Mobile UX is "good enough"

**Migrate to React if:**
- User count > 1,000
- Need professional UX
- Mobile is critical
- Want custom features
- Have budget for longer development

---

## 14. Cost Analysis

### 14.1 Project 1 (FREE)

```
Streamlit Cloud:  $0
yfinance:         $0
SQLite:           $0
GitHub:           $0
Domain (optional): $12/year

Total: $0-12/year
```

### 14.2 Project 3 (Paid)

**Minimal Viable Budget:**
```
Hosting (Streamlit Pro):     $20/month
Database (Supabase):         $25/month
Email (SendGrid):            $15/month
Auth (Clerk):                $25/month
Domain:                      $12/year

Total: ~$90/month + $12/year
```

**Professional Stack:**
```
Hosting (Vercel Pro):        $20/month
Backend (Railway):           $20/month
Database (PostgreSQL):       $25/month
Auth (Auth0):                $25/month
Analytics (Mixpanel):        $25/month
Error Tracking (Sentry):     $26/month
Email (SendGrid):            $15/month

Total: ~$160/month
```

---

## 15. Alternatives Considered

### 15.1 Why NOT...

**React + FastAPI (Initially)?**
- ❌ 3x longer development (3 weeks vs 1 week)
- ❌ Need to learn React (if not familiar)
- ❌ More complex deployment
- ✅ Better for Project 3, not Week 1

**Dash (Plotly)?**
- ❌ More verbose than Streamlit
- ❌ Steeper learning curve
- ✅ Better than Streamlit for complex apps
- ✅ Consider for Project 3

**Flask/Django?**
- ❌ Need to build UI from scratch
- ❌ No built-in components
- ✅ More control than Streamlit
- ✅ Consider if migrating from Streamlit

**Jupyter Notebook?**
- ❌ Not a web app
- ❌ Not mobile-friendly
- ✅ Good for prototyping
- ✅ Use for research, not production

---

## 16. Decision Matrix

### 16.1 Framework Selection Criteria

| Criteria | Weight | Streamlit | React | Flask | Dash |
|----------|--------|-----------|-------|-------|------|
| Development Speed | 40% | 10 | 5 | 7 | 8 |
| Mobile UX | 20% | 7 | 10 | 8 | 7 |
| Python Integration | 20% | 10 | 3 | 10 | 10 |
| Customization | 10% | 5 | 10 | 9 | 7 |
| Learning Curve | 10% | 10 | 4 | 7 | 6 |
| **Total Score** | | **8.9** | **6.1** | **7.9** | **7.9** |

**Winner for Project 1: Streamlit (8.9/10)**

---

## 17. Dependencies (requirements.txt)

```txt
# Core Framework
streamlit==1.28.0

# Data & Math
pandas==2.1.0
numpy==1.24.0
scipy==1.11.0

# Data Acquisition
yfinance==0.2.28
requests==2.31.0

# Visualization
plotly==5.17.0
matplotlib==3.8.0

# Utilities
python-dateutil==2.8.2
pytz==2023.3
tqdm==4.66.1

# Testing (dev only)
pytest==7.4.2
pytest-cov==4.1.0

# Code Quality (dev only)
black==23.9.1
pylint==3.0.0
mypy==1.5.1
```

**Install Command:**
```bash
pip install -r requirements.txt
```

---

## 18. Final Recommendation

### 18.1 For Week 1 (Project 1)

**Go with:**
```
Frontend:  Streamlit
Backend:   Pure Python
Data:      yfinance
Storage:   SQLite
Deploy:    Streamlit Cloud
```

**Why:**
- ✅ Can build in 1 week
- ✅ Achieves all learning objectives
- ✅ Mobile-responsive (good enough)
- ✅ $0 cost
- ✅ Easy to demo to class

### 18.2 For Course Project (Project 2)

**Same stack, add:**
```python
import FinanceDataReader as fdr
from pykrx import stock
```

### 18.3 For Commercial Product (Project 3)

**Evaluate then:**
- If users love it as-is → Stick with Streamlit
- If need better mobile → Migrate to Next.js
- If need complex features → FastAPI backend

---

**Document Status:** ✅ Final  
**Next Action:** Run `pip install -r requirements.txt`  
**Owner:** You (Full Stack)  
**Related Docs:** `PRD.md`, `IMPLEMENTATION_PLAN.md`
