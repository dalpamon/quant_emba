# Implementation Plan
## Factor Lab - 7-Day Build Schedule

**Project:** Factor Lab (Project 1 MVP)  
**Timeline:** October 25 - October 31, 2024  
**Sprint Goal:** Functional multi-factor backtesting platform  
**Daily Commitment:** 4-6 hours/day  

---

## 🎯 Week Overview

```
Day 1 (Fri Oct 25): Project setup + Data pipeline      [Foundation]
Day 2 (Sat Oct 26): Factor engine                      [Core Logic]
Day 3 (Sun Oct 27): Portfolio & Backtest engine        [Core Logic]
Day 4 (Mon Oct 28): Streamlit UI - Strategy Builder    [Frontend]
Day 5 (Tue Oct 29): Streamlit UI - Results Dashboard   [Frontend]
Day 6 (Wed Oct 30): Polish + Testing + Bug fixes       [Quality]
Day 7 (Thu Oct 31): Deploy + Documentation             [Launch]
```

**Sprint Success Criteria:**
- ✅ 5 factors implemented and working
- ✅ Backtest produces accurate Sharpe ratios
- ✅ Mobile-responsive UI
- ✅ Deployed to Streamlit Cloud
- ✅ Portfolio ready for course

---

## 📅 Day 1: Project Setup & Data Pipeline

**Goal:** Get data flowing from yfinance to SQLite

**Time Allocation:** 4-5 hours
- Setup: 1 hour
- Data loader: 2 hours
- Testing: 1 hour
- Documentation: 30 min

### Morning (2 hours) - Project Setup

**Task 1.1: Create Project Structure**
```bash
# Create directory
mkdir factor-lab
cd factor-lab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create folder structure
mkdir -p data/cache logs tests
mkdir -p core components pages
touch app.py README.md requirements.txt
touch core/__init__.py
```

**Task 1.2: Install Dependencies**
```bash
# Create requirements.txt
cat > requirements.txt << EOF
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.0
scipy==1.11.0
yfinance==0.2.28
plotly==5.17.0
matplotlib==3.8.0
python-dateutil==2.8.2
pytest==7.4.2
EOF

# Install
pip install -r requirements.txt
```

**Task 1.3: Git Initialize**
```bash
git init
git add .
git commit -m "Initial project setup"

# Create GitHub repo and push
git remote add origin https://github.com/yourusername/factor-lab.git
git push -u origin main
```

**✅ Checkpoint 1:** Project structure ready, dependencies installed

---

### Afternoon (3 hours) - Data Pipeline

**Task 1.4: Build Data Loader**

Create `core/data_loader.py`:

```python
"""
Data loading and caching module
Downloads stock data from yfinance and caches locally
"""

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data download and caching"""
    
    def __init__(self, db_path='data/cache/factorlab.db'):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._initialize_db()
    
    def _initialize_db(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Price data table
        cursor.execute('''
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
        
        # Create index
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ticker_date 
            ON price_data(ticker, date)
        ''')
        
        self.conn.commit()
        logger.info("Database initialized")
    
    def fetch_data(self, tickers, start_date, end_date, force_refresh=False):
        """
        Fetch stock data with caching
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download data
        
        Returns:
            pd.DataFrame: Multi-index (date, ticker) with OHLCV data
        """
        # Try cache first
        if not force_refresh:
            cached = self._load_from_cache(tickers, start_date, end_date)
            if cached is not None and not cached.empty:
                logger.info(f"✅ Loaded {len(tickers)} tickers from cache")
                return cached
        
        # Download from yfinance
        logger.info(f"📥 Downloading {len(tickers)} tickers from {start_date} to {end_date}")
        
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,
                threads=True,
                progress=False
            )
            
            if data.empty:
                logger.error("❌ No data downloaded")
                return pd.DataFrame()
            
            # Save to cache
            self._save_to_cache(data, tickers)
            logger.info(f"✅ Downloaded and cached {len(tickers)} tickers")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return pd.DataFrame()
    
    def _load_from_cache(self, tickers, start_date, end_date):
        """Load data from SQLite cache"""
        try:
            placeholders = ','.join(['?'] * len(tickers))
            query = f"""
                SELECT ticker, date, open, high, low, close, volume, adjusted_close
                FROM price_data
                WHERE ticker IN ({placeholders})
                AND date BETWEEN ? AND ?
                ORDER BY date, ticker
            """
            
            df = pd.read_sql_query(
                query,
                self.conn,
                params=tickers + [start_date, end_date],
                parse_dates=['date']
            )
            
            if df.empty:
                return None
            
            # Reshape to yfinance format (multi-level columns)
            df_pivot = df.pivot(index='date', columns='ticker')
            
            return df_pivot
            
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None
    
    def _save_to_cache(self, data, tickers):
        """Save data to SQLite cache"""
        try:
            # Flatten multi-level columns for storage
            records = []
            
            for ticker in tickers:
                if isinstance(data.columns, pd.MultiIndex):
                    ticker_data = data[ticker]
                else:
                    ticker_data = data
                
                for date, row in ticker_data.iterrows():
                    records.append({
                        'ticker': ticker,
                        'date': date.strftime('%Y-%m-%d'),
                        'open': row.get('Open', None),
                        'high': row.get('High', None),
                        'low': row.get('Low', None),
                        'close': row.get('Close', None),
                        'volume': row.get('Volume', None),
                        'adjusted_close': row.get('Close', None)
                    })
            
            # Save to DB (replace if exists)
            df = pd.DataFrame(records)
            df.to_sql('price_data', self.conn, if_exists='append', index=False)
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
    
    def get_universe(self, universe_name):
        """Get predefined stock universes"""
        universes = {
            'sp500_sample': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
                           'TSLA', 'NVDA', 'JPM', 'V', 'WMT'],
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA'],
            'faang': ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']
        }
        return universes.get(universe_name, [])


# Test function
if __name__ == "__main__":
    loader = DataLoader()
    tickers = loader.get_universe('tech')
    data = loader.fetch_data(tickers, '2020-01-01', '2024-10-25')
    print(f"Downloaded {len(data)} days of data for {len(tickers)} stocks")
    print(data.head())
```

**Task 1.5: Test Data Pipeline**

```bash
# Test the data loader
python core/data_loader.py

# Expected output:
# 📥 Downloading 7 tickers from 2020-01-01 to 2024-10-25
# ✅ Downloaded and cached 7 tickers
# Downloaded 1234 days of data for 7 stocks
```

**✅ Checkpoint 2:** Data pipeline working, can download and cache data

---

### Evening (1 hour) - Day 1 Wrap-up

**Task 1.6: Write Tests**

Create `tests/test_data_loader.py`:

```python
import pytest
from core.data_loader import DataLoader
import pandas as pd

def test_data_loader_initialization():
    """Test DataLoader creates database"""
    loader = DataLoader(db_path='data/cache/test.db')
    assert loader.conn is not None

def test_fetch_data():
    """Test downloading data"""
    loader = DataLoader()
    data = loader.fetch_data(['AAPL'], '2024-01-01', '2024-10-25')
    
    assert not data.empty
    assert len(data) > 200  # At least 200 trading days

def test_get_universe():
    """Test predefined universes"""
    loader = DataLoader()
    tech = loader.get_universe('tech')
    
    assert len(tech) == 7
    assert 'AAPL' in tech
```

**Task 1.7: Update README**

```markdown
# Factor Lab

Multi-factor quantitative investment backtesting platform.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Day 1 Progress
- ✅ Project structure
- ✅ Data pipeline (yfinance + SQLite)
- ✅ Caching implemented
```

**Task 1.8: Commit Progress**

```bash
git add .
git commit -m "Day 1: Data pipeline complete"
git push
```

**🏁 Day 1 Complete!** You now have a working data pipeline.

---

## 📅 Day 2: Factor Engine

**Goal:** Calculate 5 quantitative factors

**Time Allocation:** 5-6 hours
- Factor implementations: 3 hours
- Testing: 1 hour
- Documentation: 1 hour

### Morning (3 hours) - Factor Calculations

**Task 2.1: Create Factor Engine**

Create `core/factor_engine.py`:

```python
"""
Factor calculation engine
Computes quantitative factors from price data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class FactorEngine:
    """Calculate quantitative factors"""
    
    @staticmethod
    def calculate_all_factors(prices, tickers=None):
        """
        Calculate all factors
        
        Args:
            prices: DataFrame with adjusted close prices (date x ticker)
            tickers: List of tickers (for fundamentals)
        
        Returns:
            DataFrame: Factor scores (date x ticker x factor)
        """
        logger.info("📊 Calculating factors...")
        
        factors = pd.DataFrame()
        
        # Price-based factors (fast)
        factors['momentum'] = FactorEngine._momentum(prices)
        factors['volatility'] = FactorEngine._volatility(prices)
        
        # Fundamental factors (slower, need to fetch)
        if tickers:
            fundamentals = FactorEngine._get_fundamentals(tickers)
            factors['value'] = fundamentals['value']
            factors['size'] = fundamentals['size']
            factors['quality'] = fundamentals['quality']
        
        logger.info(f"✅ Calculated {len(factors.columns)} factors")
        return factors
    
    @staticmethod
    def _momentum(prices, lookback=252, skip_last=21):
        """
        12-month momentum (skip last month)
        
        Academic basis: Jegadeesh & Titman (1993)
        """
        if len(prices) < lookback + skip_last:
            logger.warning("Not enough data for momentum calculation")
            return pd.Series(np.nan, index=prices.columns)
        
        # Calculate returns from 252 days ago to 21 days ago
        start_price = prices.iloc[-(lookback + skip_last)]
        end_price = prices.iloc[-skip_last]
        
        momentum = (end_price / start_price - 1)
        return momentum
    
    @staticmethod
    def _volatility(prices, window=60):
        """
        60-day volatility (lower is better)
        
        Academic basis: Low-volatility anomaly (Ang et al., 2006)
        """
        if len(prices) < window:
            logger.warning("Not enough data for volatility calculation")
            return pd.Series(np.nan, index=prices.columns)
        
        returns = prices.pct_change().iloc[-window:]
        volatility = returns.std()
        
        # Negative because lower vol is better
        return -volatility
    
    @staticmethod
    def _get_fundamentals(tickers):
        """
        Fetch fundamental data for value, size, quality
        """
        fundamentals = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                market_cap = info.get('marketCap', np.nan)
                pe_ratio = info.get('trailingPE', np.nan)
                pb_ratio = info.get('priceToBook', np.nan)
                profit_margin = info.get('profitMargins', np.nan)
                
                # Value: Inverse P/B (higher is better)
                value = 1 / pb_ratio if pb_ratio and pb_ratio > 0 else np.nan
                
                # Size: Log market cap (smaller is better in small-cap premium)
                size = -np.log(market_cap) if market_cap and market_cap > 0 else np.nan
                
                # Quality: Profit margin
                quality = profit_margin if profit_margin else np.nan
                
                fundamentals.append({
                    'ticker': ticker,
                    'value': value,
                    'size': size,
                    'quality': quality
                })
                
            except Exception as e:
                logger.warning(f"Failed to get fundamentals for {ticker}: {e}")
                fundamentals.append({
                    'ticker': ticker,
                    'value': np.nan,
                    'size': np.nan,
                    'quality': np.nan
                })
        
        df = pd.DataFrame(fundamentals).set_index('ticker')
        return df
    
    @staticmethod
    def normalize_factors(factors):
        """
        Z-score normalization for cross-sectional comparison
        
        Args:
            factors: DataFrame of raw factor scores
        
        Returns:
            DataFrame: Normalized factor scores (mean=0, std=1)
        """
        normalized = factors.copy()
        
        for col in factors.columns:
            mean = factors[col].mean()
            std = factors[col].std()
            
            if std > 0:
                normalized[col] = (factors[col] - mean) / std
            else:
                normalized[col] = 0
        
        return normalized
    
    @staticmethod
    def combine_factors(factors, weights):
        """
        Create composite score from weighted factors
        
        Args:
            factors: DataFrame of normalized factors
            weights: Dict like {'momentum': 0.4, 'value': 0.3, ...}
        
        Returns:
            Series: Composite scores
        """
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        composite = pd.Series(0.0, index=factors.index)
        
        for factor, weight in weights.items():
            if factor in factors.columns:
                composite += weight * factors[factor]
        
        return composite


# Test
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    tickers = loader.get_universe('tech')
    data = loader.fetch_data(tickers, '2020-01-01', '2024-10-25')
    
    # Extract close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs('Close', level=1, axis=1)
    else:
        prices = data['Close']
    
    # Calculate factors
    engine = FactorEngine()
    factors = engine.calculate_all_factors(prices, tickers)
    
    print("\nRaw Factors:")
    print(factors)
    
    # Normalize
    normalized = engine.normalize_factors(factors)
    print("\nNormalized Factors:")
    print(normalized)
    
    # Combine
    weights = {'momentum': 0.4, 'value': 0.3, 'quality': 0.3}
    composite = engine.combine_factors(normalized, weights)
    print("\nComposite Scores:")
    print(composite)
```

**✅ Checkpoint 3:** Factor engine complete, can calculate 5 factors

---

### Afternoon (2 hours) - Testing & Validation

**Task 2.2: Test Factors**

```bash
python core/factor_engine.py

# Verify:
# - All 5 factors calculated
# - No NaN values (or handled gracefully)
# - Normalized factors have mean ≈ 0, std ≈ 1
```

**Task 2.3: Write Factor Tests**

Create `tests/test_factor_engine.py`:

```python
import pytest
import pandas as pd
import numpy as np
from core.factor_engine import FactorEngine

def test_momentum_calculation():
    """Test momentum returns correct values"""
    # Create fake price series: 100 to 120 over 252 days
    dates = pd.date_range('2023-01-01', periods=300)
    prices = pd.DataFrame({
        'AAPL': np.linspace(100, 120, 300)
    }, index=dates)
    
    momentum = FactorEngine._momentum(prices, lookback=252, skip_last=21)
    
    # Should be approximately +20% (120/100 - 1)
    assert abs(momentum['AAPL'] - 0.20) < 0.05

def test_normalize_factors():
    """Test normalization creates z-scores"""
    factors = pd.DataFrame({
        'momentum': [0.1, 0.2, 0.3, 0.4, 0.5],
        'value': [1.0, 2.0, 3.0, 4.0, 5.0]
    }, index=['A', 'B', 'C', 'D', 'E'])
    
    normalized = FactorEngine.normalize_factors(factors)
    
    # Check mean ≈ 0, std ≈ 1
    assert abs(normalized['momentum'].mean()) < 0.0001
    assert abs(normalized['momentum'].std() - 1.0) < 0.0001

def test_combine_factors():
    """Test weighted combination"""
    factors = pd.DataFrame({
        'momentum': [1.0, -1.0, 0.0],
        'value': [0.5, 0.5, -1.0]
    }, index=['A', 'B', 'C'])
    
    weights = {'momentum': 0.6, 'value': 0.4}
    composite = FactorEngine.combine_factors(factors, weights)
    
    # A: 0.6*1.0 + 0.4*0.5 = 0.8
    assert abs(composite['A'] - 0.8) < 0.001
```

**Task 2.4: Commit Day 2**

```bash
pytest tests/test_factor_engine.py
git add .
git commit -m "Day 2: Factor engine complete with 5 factors"
git push
```

**🏁 Day 2 Complete!** You can now calculate factors.

---

## 📅 Day 3: Portfolio & Backtest Engine

**Goal:** Build portfolio construction and backtesting

**Time Allocation:** 5-6 hours

### Morning (3 hours) - Portfolio Constructor

**Task 3.1: Create Portfolio Builder**

Create `core/portfolio.py`:

```python
"""
Portfolio construction module
Long-short portfolio based on factor scores
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PortfolioBuilder:
    """Construct long-short portfolios from factor scores"""
    
    def __init__(self, top_pct=0.2, bottom_pct=0.2):
        """
        Args:
            top_pct: Top X% to go long
            bottom_pct: Bottom X% to go short
        """
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
    
    def construct_long_short(self, scores):
        """
        Create long-short portfolio from composite scores
        
        Args:
            scores: Series of composite factor scores
        
        Returns:
            Dict with 'long' and 'short' lists
        """
        # Remove NaN values
        scores = scores.dropna()
        
        if len(scores) == 0:
            logger.warning("No valid scores for portfolio construction")
            return {'long': [], 'short': []}
        
        # Calculate thresholds
        long_threshold = scores.quantile(1 - self.top_pct)
        short_threshold = scores.quantile(self.bottom_pct)
        
        # Select stocks
        long_stocks = scores[scores >= long_threshold].index.tolist()
        short_stocks = scores[scores <= short_threshold].index.tolist()
        
        logger.info(f"Portfolio: {len(long_stocks)} long, {len(short_stocks)} short")
        
        return {
            'long': long_stocks,
            'short': short_stocks
        }
    
    def get_weights(self, positions):
        """
        Calculate equal weights for positions
        
        Args:
            positions: Dict with 'long' and 'short' lists
        
        Returns:
            Dict: {ticker: weight}
        """
        weights = {}
        
        n_long = len(positions['long'])
        n_short = len(positions['short'])
        
        # Equal weight within long and short
        if n_long > 0:
            long_weight = 0.5 / n_long  # 50% total in long
            for ticker in positions['long']:
                weights[ticker] = long_weight
        
        if n_short > 0:
            short_weight = -0.5 / n_short  # 50% total in short
            for ticker in positions['short']:
                weights[ticker] = short_weight
        
        return weights


if __name__ == "__main__":
    # Test
    scores = pd.Series({
        'AAPL': 1.5,
        'MSFT': 1.2,
        'GOOGL': 0.8,
        'AMZN': 0.3,
        'META': -0.2,
        'TSLA': -1.0,
        'NVDA': -1.5
    })
    
    builder = PortfolioBuilder(top_pct=0.3, bottom_pct=0.3)
    portfolio = builder.construct_long_short(scores)
    weights = builder.get_weights(portfolio)
    
    print("Portfolio:")
    print(portfolio)
    print("\nWeights:")
    print(weights)
```

### Afternoon (3 hours) - Backtest Engine

**Task 3.2: Create Backtester**

Create `core/backtester.py`:

```python
"""
Backtesting engine
Simulates portfolio performance over time
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Backtester:
    """Vectorized backtesting engine"""
    
    def __init__(self, initial_capital=100000, rebalance_freq='M', transaction_cost_bps=10):
        """
        Args:
            initial_capital: Starting portfolio value
            rebalance_freq: 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
            transaction_cost_bps: Transaction cost in basis points (10 bps = 0.1%)
        """
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost_bps / 10000
    
    def run(self, prices, positions_by_date):
        """
        Run backtest
        
        Args:
            prices: DataFrame of prices (date x ticker)
            positions_by_date: Dict {date: {'long': [...], 'short': [...]}}
        
        Returns:
            DataFrame: Daily portfolio value
        """
        logger.info(f"🚀 Running backtest from {prices.index[0]} to {prices.index[-1]}")
        
        # Calculate daily returns
        returns = prices.pct_change()
        
        # Initialize tracking
        portfolio_values = [self.initial_capital]
        dates = [prices.index[0]]
        
        current_long = []
        current_short = []
        
        rebalance_dates = list(positions_by_date.keys())
        
        for i, date in enumerate(prices.index[1:], 1):
            # Check if we need to rebalance
            if date in rebalance_dates:
                current_long = positions_by_date[date]['long']
                current_short = positions_by_date[date]['short']
                
                # Apply transaction cost on rebalance
                turnover = 1.0  # Assume 100% turnover on rebalance
                cost = portfolio_values[-1] * self.transaction_cost * turnover
                portfolio_values[-1] -= cost
            
            # Calculate portfolio return for the day
            if current_long or current_short:
                # Long portfolio return
                long_return = 0
                if current_long:
                    long_stocks_returns = returns.loc[date, current_long]
                    long_return = long_stocks_returns.mean() if not pd.isna(long_stocks_returns).all() else 0
                
                # Short portfolio return (negative of stock returns)
                short_return = 0
                if current_short:
                    short_stocks_returns = returns.loc[date, current_short]
                    short_return = -short_stocks_returns.mean() if not pd.isna(short_stocks_returns).all() else 0
                
                # Combined portfolio return (50% long, 50% short)
                portfolio_return = (long_return + short_return) / 2
            else:
                portfolio_return = 0
            
            # Update portfolio value
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
            dates.append(date)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'portfolio_value': portfolio_values
        }, index=dates)
        
        logger.info(f"✅ Backtest complete. Final value: ${portfolio_values[-1]:,.0f}")
        
        return results


class PerformanceAnalytics:
    """Calculate performance metrics"""
    
    @staticmethod
    def calculate_metrics(equity_curve):
        """
        Calculate all performance metrics
        
        Args:
            equity_curve: Series of portfolio values
        
        Returns:
            Dict of metrics
        """
        returns = equity_curve.pct_change().dropna()
        
        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # CAGR
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        # Max drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Winning days
        win_rate = (returns > 0).sum() / len(returns)
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(equity_curve)
        }


# Test
if __name__ == "__main__":
    from data_loader import DataLoader
    from factor_engine import FactorEngine
    from portfolio import PortfolioBuilder
    
    # Load data
    loader = DataLoader()
    tickers = loader.get_universe('tech')
    data = loader.fetch_data(tickers, '2020-01-01', '2024-10-25')
    
    # Extract prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs('Close', level=1, axis=1)
    else:
        prices = data['Close']
    
    # Calculate factors
    factors = FactorEngine.calculate_all_factors(prices, tickers)
    normalized = FactorEngine.normalize_factors(factors)
    
    weights = {'momentum': 0.6, 'value': 0.4}
    scores = FactorEngine.combine_factors(normalized, weights)
    
    # Build portfolio (monthly rebalancing)
    builder = PortfolioBuilder()
    
    # Get month-end dates
    month_ends = prices.resample('M').last().index
    
    positions_by_date = {}
    for date in month_ends:
        if date in prices.index:
            positions_by_date[date] = builder.construct_long_short(scores)
    
    # Backtest
    backtester = Backtester(initial_capital=100000, rebalance_freq='M')
    results = backtester.run(prices, positions_by_date)
    
    # Analytics
    metrics = PerformanceAnalytics.calculate_metrics(results['portfolio_value'])
    
    print("\n📊 Performance Metrics:")
    for metric, value in metrics.items():
        if 'return' in metric or 'drawdown' in metric or 'rate' in metric:
            print(f"{metric:20s}: {value:>8.2%}")
        else:
            print(f"{metric:20s}: {value:>8.2f}")
```

**✅ Checkpoint 4:** Backtest engine complete, can simulate strategies

**Task 3.3: Integration Test**

```bash
python core/backtester.py

# Should output:
# 🚀 Running backtest...
# ✅ Backtest complete. Final value: $145,234
# 📊 Performance Metrics:
# total_return:      45.23%
# sharpe_ratio:       1.45
# ...
```

**Task 3.4: Commit Day 3**

```bash
git add .
git commit -m "Day 3: Portfolio construction & backtesting complete"
git push
```

**🏁 Day 3 Complete!** Core engine is done!

---

## 📅 Day 4: Streamlit UI - Strategy Builder

**Goal:** Build the strategy configuration interface

**Time Allocation:** 5-6 hours

### Task 4.1: Create Main App Structure

Create `app.py`:

```python
"""
Factor Lab - Main Streamlit App
"""

import streamlit as st
import sys
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Factor Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00D9FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Home page
st.markdown('<h1 class="main-header">📊 Factor Lab</h1>', unsafe_allow_html=True)
st.markdown("### Build & Test Quantitative Investment Strategies")

# Hero section
col1, col2, col3 = st.columns(3)

with col1:
    st.info("**1. Select Factors**\nChoose momentum, value, quality, and more")

with col2:
    st.info("**2. Build Strategy**\nCombine factors with custom weights")

with col3:
    st.info("**3. Analyze Results**\nView performance metrics and charts")

st.markdown("---")

# Quick start options
st.subheader("🚀 Quick Start")

col1, col2 = st.columns(2)

with col1:
    if st.button("Try Example Strategy", type="primary", use_container_width=True):
        st.switch_page("pages/1_Strategy_Builder.py")

with col2:
    if st.button("Build Your Own", use_container_width=True):
        st.switch_page("pages/1_Strategy_Builder.py")

# Educational content
with st.expander("💡 What is Factor Investing?"):
    st.write("""
    Factor investing is a strategy that targets specific drivers of return 
    across asset classes. Common factors include:
    
    - **Momentum**: Stocks that have performed well continue to perform well
    - **Value**: Cheap stocks (low P/B, P/E) outperform
    - **Size**: Small-cap stocks earn higher returns
    - **Quality**: Profitable companies with stable earnings
    - **Low Volatility**: Lower-risk stocks earn comparable returns
    """)

st.markdown("---")
st.caption("Built with Streamlit | Data from Yahoo Finance")
```

### Task 4.2: Create Strategy Builder Page

Create `pages/1_Strategy_Builder.py`:

```python
"""
Strategy Builder Page
Configure factors and portfolio settings
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.data_loader import DataLoader
from core.factor_engine import FactorEngine
from core.portfolio import PortfolioBuilder
from core.backtester import Backtester, PerformanceAnalytics

# Page config
st.set_page_config(page_title="Strategy Builder", page_icon="🔧", layout="wide")

st.title("🔧 Strategy Builder")

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# Sidebar configuration
st.sidebar.header("Configuration")

# Universe selection
universe_options = {
    'Tech (7 stocks)': 'tech',
    'S&P 500 Sample (10 stocks)': 'sp500_sample',
    'FAANG': 'faang'
}
selected_universe = st.sidebar.selectbox(
    "Stock Universe",
    options=list(universe_options.keys())
)
universe_key = universe_options[selected_universe]

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-10-25"))

st.sidebar.markdown("---")

# Factor weights
st.sidebar.header("📊 Factor Weights")

momentum_weight = st.sidebar.slider(
    "Momentum (12-month)",
    0, 100, 40,
    help="Stocks with strong past performance"
)

value_weight = st.sidebar.slider(
    "Value (P/B)",
    0, 100, 30,
    help="Cheap stocks relative to book value"
)

quality_weight = st.sidebar.slider(
    "Quality (Profitability)",
    0, 100, 30,
    help="Profitable companies"
)

size_weight = st.sidebar.slider(
    "Size (Market Cap)",
    0, 100, 0,
    help="Small-cap premium"
)

volatility_weight = st.sidebar.slider(
    "Low Volatility",
    0, 100, 0,
    help="Lower-risk stocks"
)

# Validate weights
total_weight = momentum_weight + value_weight + quality_weight + size_weight + volatility_weight

if total_weight != 100:
    st.sidebar.warning(f"⚠️ Weights sum to {total_weight}%. Auto-normalizing to 100%")

# Normalize weights
factor_weights = {
    'momentum': momentum_weight / total_weight if total_weight > 0 else 0,
    'value': value_weight / total_weight if total_weight > 0 else 0,
    'quality': quality_weight / total_weight if total_weight > 0 else 0,
    'size': size_weight / total_weight if total_weight > 0 else 0,
    'volatility': volatility_weight / total_weight if total_weight > 0 else 0
}

# Remove zero-weight factors
factor_weights = {k: v for k, v in factor_weights.items() if v > 0}

st.sidebar.markdown("---")

# Portfolio settings
st.sidebar.header("⚙️ Portfolio Settings")

portfolio_type = st.sidebar.selectbox(
    "Strategy Type",
    ["Long-Short", "Long-Only"]
)

rebalance_freq = st.sidebar.selectbox(
    "Rebalancing",
    ["Monthly", "Quarterly", "Annually"]
)

transaction_cost = st.sidebar.number_input(
    "Transaction Cost (bps)",
    min_value=0,
    max_value=100,
    value=10,
    help="Basis points per trade (10 bps = 0.1%)"
)

# Main area - display strategy summary
st.subheader("Strategy Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Universe", selected_universe)
    st.metric("Period", f"{(end_date - start_date).days} days")

with col2:
    st.metric("Active Factors", len(factor_weights))
    st.metric("Rebalancing", rebalance_freq)

with col3:
    st.metric("Strategy Type", portfolio_type)
    st.metric("Transaction Cost", f"{transaction_cost} bps")

st.markdown("---")

# Display factor weights
st.subheader("Factor Allocation")

if factor_weights:
    weights_df = pd.DataFrame(
        list(factor_weights.items()),
        columns=['Factor', 'Weight']
    )
    weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(weights_df, use_container_width=True, hide_index=True)
else:
    st.warning("⚠️ Please select at least one factor")

st.markdown("---")

# Run backtest button
if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
    
    if not factor_weights:
        st.error("❌ Please select at least one factor")
        st.stop()
    
    with st.spinner("Running backtest... This may take 30-60 seconds"):
        try:
            # Load data
            progress_bar = st.progress(0, text="Loading data...")
            
            loader = DataLoader()
            tickers = loader.get_universe(universe_key)
            data = loader.fetch_data(tickers, start_date.strftime('%Y-%m-%d'), 
                                   end_date.strftime('%Y-%m-%d'))
            
            progress_bar.progress(25, text="Calculating factors...")
            
            # Extract prices
            if isinstance(data.columns, pd.MultiIndex):
                prices = data.xs('Close', level=1, axis=1)
            else:
                prices = data['Close']
            
            # Calculate factors
            factors = FactorEngine.calculate_all_factors(prices, tickers)
            normalized = FactorEngine.normalize_factors(factors)
            
            progress_bar.progress(50, text="Building portfolio...")
            
            # Combine factors
            scores = FactorEngine.combine_factors(normalized, factor_weights)
            
            # Build portfolio
            builder = PortfolioBuilder(top_pct=0.2, bottom_pct=0.2)
            
            # Get rebalancing dates
            freq_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Annually': 'Y'}
            rebal_dates = prices.resample(freq_map[rebalance_freq]).last().index
            
            positions_by_date = {}
            for date in rebal_dates:
                if date in prices.index:
                    positions_by_date[date] = builder.construct_long_short(scores)
            
            progress_bar.progress(75, text="Running backtest...")
            
            # Backtest
            backtester = Backtester(
                initial_capital=100000,
                rebalance_freq=freq_map[rebalance_freq],
                transaction_cost_bps=transaction_cost
            )
            results = backtester.run(prices, positions_by_date)
            
            progress_bar.progress(90, text="Calculating metrics...")
            
            # Calculate metrics
            metrics = PerformanceAnalytics.calculate_metrics(results['portfolio_value'])
            
            progress_bar.progress(100, text="Complete!")
            progress_bar.empty()
            
            # Store in session state
            st.session_state.backtest_results = {
                'equity_curve': results,
                'metrics': metrics,
                'positions': positions_by_date,
                'strategy_config': {
                    'factors': factor_weights,
                    'universe': selected_universe,
                    'dates': (start_date, end_date),
                    'rebalance': rebalance_freq
                }
            }
            
            st.success("✅ Backtest complete! View results below.")
            
            # Switch to results page
            st.switch_page("pages/2_Results.py")
            
        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")
            st.exception(e)

# Show example if no backtest run yet
if st.session_state.backtest_results is None:
    st.info("👈 Configure your strategy in the sidebar and click 'Run Backtest'")
```

**✅ Checkpoint 5:** Strategy Builder UI complete

**Task 4.3: Test UI**

```bash
streamlit run app.py

# Test:
# 1. Navigate to Strategy Builder
# 2. Adjust factor sliders
# 3. Click "Run Backtest"
# 4. Verify it completes without errors
```

**Task 4.4: Commit Day 4**

```bash
git add .
git commit -m "Day 4: Strategy Builder UI complete"
git push
```

**🏁 Day 4 Complete!** UI is functional!

---

## 📅 Day 5: Results Dashboard

**Goal:** Display backtest results with interactive charts

**Time Allocation:** 5-6 hours

### Morning (3 hours) - Results Page

**Task 5.1: Create Results Page**

Create `pages/2_Results.py`:

```python
"""
Results Dashboard
Display backtest performance metrics and visualizations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")

st.title("📈 Backtest Results")

# Check if results exist
if 'backtest_results' not in st.session_state or st.session_state.backtest_results is None:
    st.warning("⚠️ No backtest results available. Please run a backtest first.")
    if st.button("Go to Strategy Builder"):
        st.switch_page("pages/1_Strategy_Builder.py")
    st.stop()

# Load results
results = st.session_state.backtest_results
equity_curve = results['equity_curve']['portfolio_value']
metrics = results['metrics']
config = results['strategy_config']

# Strategy info header
st.subheader(f"Strategy: {' + '.join(config['factors'].keys()).title()}")
st.caption(f"Universe: {config['universe']} | Period: {config['dates'][0]} to {config['dates'][1]} | Rebalancing: {config['rebalance']}")

st.markdown("---")

# Key Metrics (Cards)
st.subheader("📊 Performance Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_return = metrics['total_return']
    st.metric(
        "Total Return",
        f"{total_return:.1%}",
        delta=f"{total_return:.1%}",
        delta_color="normal"
    )

with col2:
    cagr = metrics['cagr']
    st.metric(
        "CAGR",
        f"{cagr:.1%}",
        help="Compound Annual Growth Rate"
    )

with col3:
    sharpe = metrics['sharpe_ratio']
    sharpe_color = "normal" if sharpe > 1 else "inverse"
    st.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}",
        delta="Good" if sharpe > 1 else "Poor",
        delta_color=sharpe_color,
        help="Risk-adjusted return (>1 is good)"
    )

with col4:
    max_dd = metrics['max_drawdown']
    st.metric(
        "Max Drawdown",
        f"{max_dd:.1%}",
        delta=f"{max_dd:.1%}",
        delta_color="inverse"
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    vol = metrics['volatility']
    st.metric("Volatility", f"{vol:.1%}")

with col2:
    sortino = metrics['sortino_ratio']
    st.metric("Sortino Ratio", f"{sortino:.2f}", help="Downside risk-adjusted return")

with col3:
    win_rate = metrics['win_rate']
    st.metric("Win Rate", f"{win_rate:.1%}")

with col4:
    num_trades = metrics['num_trades']
    st.metric("Days", f"{num_trades:,}")

st.markdown("---")

# Equity Curve Chart
st.subheader("📈 Equity Curve")

fig_equity = go.Figure()

fig_equity.add_trace(go.Scatter(
    x=equity_curve.index,
    y=equity_curve.values,
    mode='lines',
    name='Portfolio Value',
    line=dict(color='#00D9FF', width=2),
    fill='tonexty',
    fillcolor='rgba(0, 217, 255, 0.1)',
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: $%{y:,.0f}<extra></extra>'
))

fig_equity.update_layout(
    title="Portfolio Growth Over Time",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode='x unified',
    height=400,
    template='plotly_white',
    showlegend=True
)

st.plotly_chart(fig_equity, use_container_width=True)

st.markdown("---")

# Tabs for additional analysis
tab1, tab2, tab3, tab4 = st.tabs(["📊 Returns", "📉 Drawdown", "📅 Periodic Returns", "ℹ️ Details"])

with tab1:
    st.subheader("Returns Distribution")
    
    returns = equity_curve.pct_change().dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Daily Returns',
            marker_color='#00D9FF'
        ))
        fig_hist.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Return",
            yaxis_title="Frequency",
            showlegend=False,
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Stats table
        st.write("**Returns Statistics**")
        returns_stats = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{returns.mean():.4%}",
                f"{returns.median():.4%}",
                f"{returns.std():.4%}",
                f"{returns.skew():.2f}",
                f"{returns.kurtosis():.2f}"
            ]
        })
        st.dataframe(returns_stats, hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Drawdown Analysis")
    
    # Calculate drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    # Drawdown chart
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#F44336'),
        fillcolor='rgba(244, 67, 54, 0.3)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2%}<extra></extra>'
    ))
    fig_dd.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        height=350,
        template='plotly_white',
        showlegend=False
    )
    fig_dd.update_yaxes(tickformat='.0%')
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Drawdown stats
    st.write("**Drawdown Statistics**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Drawdown", f"{drawdown.min():.2%}")
    with col2:
        # Find longest drawdown period
        underwater = (drawdown < 0).astype(int)
        underwater_periods = underwater.groupby((underwater != underwater.shift()).cumsum()).sum()
        longest_dd = underwater_periods.max() if len(underwater_periods) > 0 else 0
        st.metric("Longest Drawdown", f"{longest_dd} days")
    with col3:
        current_dd = drawdown.iloc[-1]
        st.metric("Current Drawdown", f"{current_dd:.2%}")

with tab3:
    st.subheader("Periodic Returns")
    
    # Calculate monthly returns
    monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
    
    # Create year-month columns
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    # Pivot for heatmap
    heatmap_data = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')
    heatmap_data.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Heatmap
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f'{val:.1%}' for val in row] for row in heatmap_data.values],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Return", tickformat='.0%')
    ))
    
    fig_heat.update_layout(
        title="Monthly Returns Heatmap",
        xaxis_title="Year",
        yaxis_title="Month",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Annual returns table
    annual_returns = equity_curve.resample('Y').last().pct_change().dropna()
    annual_returns_df = pd.DataFrame({
        'Year': annual_returns.index.year,
        'Return': annual_returns.values
    })
    annual_returns_df['Return'] = annual_returns_df['Return'].apply(lambda x: f'{x:.2%}')
    
    st.write("**Annual Returns**")
    st.dataframe(annual_returns_df, hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Strategy Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Factor Weights**")
        factors_df = pd.DataFrame(
            list(config['factors'].items()),
            columns=['Factor', 'Weight']
        )
        factors_df['Weight'] = factors_df['Weight'].apply(lambda x: f'{x:.1%}')
        st.dataframe(factors_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.write("**Configuration**")
        config_df = pd.DataFrame({
            'Setting': ['Universe', 'Start Date', 'End Date', 'Rebalancing'],
            'Value': [
                config['universe'],
                config['dates'][0].strftime('%Y-%m-%d'),
                config['dates'][1].strftime('%Y-%m-%d'),
                config['rebalance']
            ]
        })
        st.dataframe(config_df, hide_index=True, use_container_width=True)
    
    st.write("**All Metrics**")
    all_metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': [f'{v:.2%}' if 'return' in k or 'rate' in k or 'drawdown' in k 
                 else f'{v:.2f}' for k, v in metrics.items()]
    })
    st.dataframe(all_metrics_df, hide_index=True, use_container_width=True)

st.markdown("---")

# Action buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Run New Backtest", use_container_width=True):
        st.switch_page("pages/1_Strategy_Builder.py")

with col2:
    # Export results button
    if st.button("📥 Export Results (JSON)", use_container_width=True):
        import json
        
        export_data = {
            'strategy_config': {
                'factors': config['factors'],
                'universe': config['universe'],
                'rebalancing': config['rebalance']
            },
            'metrics': metrics,
            'equity_curve': equity_curve.to_dict()
        }
        
        json_str = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="backtest_results.json",
            mime="application/json"
        )

with col3:
    # Export equity curve as CSV
    csv = equity_curve.to_csv()
    st.download_button(
        label="📥 Export Equity Curve (CSV)",
        data=csv,
        file_name="equity_curve.csv",
        mime="text/csv",
        use_container_width=True
    )
```

**✅ Checkpoint 6:** Results dashboard complete with charts

### Afternoon (2 hours) - Factor Explorer (Educational)

**Task 5.2: Create Factor Explorer**

Create `pages/3_Factor_Explorer.py`:

```python
"""
Factor Explorer - Educational page about each factor
"""

import streamlit as st

st.set_page_config(page_title="Factor Explorer", page_icon="🔍", layout="wide")

st.title("🔍 Factor Explorer")
st.write("Learn about quantitative factors and their academic foundations")

st.markdown("---")

# Factor tabs
factor_tabs = st.tabs(["📊 Momentum", "💰 Value", "⭐ Quality", "📏 Size", "📉 Low Volatility"])

with factor_tabs[0]:  # Momentum
    st.header("📊 Momentum Factor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Definition")
        st.write("""
        The momentum factor captures the tendency of stocks that have performed well 
        in the past to continue performing well in the near future, and vice versa.
        """)
        
        st.subheader("Calculation")
        st.code("""
# 12-month momentum (skip last month)
momentum = (Price_t / Price_t-252) - 1

# Skip last month to avoid reversal effects
momentum = momentum.shift(21 days)
        """, language="python")
        
        st.subheader("Academic Research")
        st.write("""
        **Jegadeesh & Titman (1993)** "Returns to Buying Winners and Selling Losers"
        
        - Found that strategies buying past winners and selling past losers generate 
          significant abnormal returns
        - Effect persists for 3-12 months
        - Strongest with 6-12 month formation period
        """)
        
        st.subheader("Why It Works")
        st.write("""
        **Behavioral Explanation:**
        - Under-reaction to news (slow incorporation of information)
        - Herding behavior (investors follow trends)
        
        **Risk-Based Explanation:**
        - Compensation for bearing systematic risk
        - Winners become riskier over time
        """)
    
    with col2:
        st.subheader("Key Characteristics")
        
        st.metric("Typical Return", "8-12% annually")
        st.metric("Sharpe Ratio", "0.6 - 0.8")
        st.metric("Best Period", "3-12 months")
        
        st.info("**Works Best In:**\n- Bull markets\n- Low volatility periods\n- Trending markets")
        
        st.warning("**Risks:**\n- Crashes (momentum crashes)\n- Reversals\n- High turnover")

with factor_tabs[1]:  # Value
    st.header("💰 Value Factor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Definition")
        st.write("""
        The value factor identifies stocks trading below their intrinsic value, 
        typically measured by Price-to-Book, Price-to-Earnings, or other valuation ratios.
        """)
        
        st.subheader("Calculation")
        st.code("""
# Book-to-Market ratio
value_score = 1 / (Price / Book_Value_Per_Share)

# Or Price-to-Earnings
value_score = 1 / PE_Ratio

# Higher score = cheaper = more value
        """, language="python")
        
        st.subheader("Academic Research")
        st.write("""
        **Fama & French (1993)** "Common Risk Factors in Returns"
        
        - Value stocks (high book-to-market) outperform growth stocks
        - Effect is stronger for small-cap stocks
        - Average value premium: ~5% per year
        """)
    
    with col2:
        st.metric("Value Premium", "4-6% annually")
        st.metric("Sharpe Ratio", "0.3 - 0.5")
        
        st.info("**Works Best In:**\n- Recovery periods\n- Mean reversion\n- Economic expansions")

with factor_tabs[2]:  # Quality
    st.header("⭐ Quality Factor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Definition")
        st.write("""
        Quality stocks are characterized by high profitability, stable earnings, 
        low debt, and efficient operations.
        """)
        
        st.subheader("Calculation")
        st.code("""
# Return on Assets (ROA)
quality_score = Net_Income / Total_Assets

# Or Return on Equity
quality_score = Net_Income / Shareholders_Equity

# Or Profit Margin
quality_score = Net_Income / Revenue
        """, language="python")
        
        st.subheader("Academic Research")
        st.write("""
        **Novy-Marx (2013)** "The Other Side of Value: Gross Profitability Premium"
        
        - Profitable firms earn higher returns than unprofitable firms
        - Quality is distinct from value and momentum
        - Works across all market caps
        """)
    
    with col2:
        st.metric("Quality Premium", "3-5% annually")
        st.metric("Sharpe Ratio", "0.5 - 0.7")
        
        st.info("**Works Best In:**\n- Market downturns\n- Defensive periods\n- Flight to safety")

with factor_tabs[3]:  # Size
    st.header("📏 Size Factor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Definition")
        st.write("""
        The size factor captures the tendency of small-cap stocks to outperform 
        large-cap stocks over the long term.
        """)
        
        st.subheader("Calculation")
        st.code("""
# Log market cap (smaller is better)
size_score = -log(Market_Capitalization)

# Or use percentile ranking
size_score = percentile_rank(Market_Cap)
        """, language="python")
        
        st.subheader("Academic Research")
        st.write("""
        **Banz (1981)** "The Relationship Between Return and Market Value"
        
        - Small stocks outperformed large stocks historically
        - Size premium has weakened since discovery
        - Still significant in international markets
        """)
    
    with col2:
        st.metric("Size Premium", "2-3% annually")
        st.metric("Sharpe Ratio", "0.2 - 0.4")
        
        st.warning("**Note:**\n- Weaker in recent decades\n- Higher in small-cap universe\n- Illiquid")

with factor_tabs[4]:  # Low Volatility
    st.header("📉 Low Volatility Factor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Definition")
        st.write("""
        The low volatility anomaly shows that stocks with lower volatility 
        earn comparable or higher returns than high volatility stocks.
        """)
        
        st.subheader("Calculation")
        st.code("""
# 60-day rolling volatility
volatility = returns.rolling(60).std()

# Lower volatility = higher score
low_vol_score = -volatility

# Or use beta
low_vol_score = -beta
        """, language="python")
        
        st.subheader("Academic Research")
        st.write("""
        **Ang et al. (2006)** "The Cross-Section of Volatility and Expected Returns"
        
        - Stocks with high idiosyncratic volatility have low returns
        - Contradicts traditional risk-return tradeoff
        - Robust across markets and time periods
        """)
    
    with col2:
        st.metric("Low-Vol Premium", "3-4% annually")
        st.metric("Sharpe Ratio", "0.7 - 1.0")
        
        st.info("**Works Best In:**\n- Bear markets\n- High uncertainty\n- Risk-off periods")

st.markdown("---")

st.subheader("📚 Recommended Reading")

st.write("""
**Books:**
- "Your Complete Guide to Factor-Based Investing" by Andrew Berkin
- "Quantitative Momentum" by Wesley Gray
- "Deep Value" by Tobias Carlisle

**Papers:**
- Fama & French (1993) - Three-Factor Model
- Carhart (1997) - Four-Factor Model  
- Fama & French (2015) - Five-Factor Model

**Websites:**
- Alpha Architect (alphaarchitect.com)
- AQR Capital Management Research
- SSRN Quantitative Finance Papers
""")
```

**✅ Checkpoint 7:** Educational content complete

**Task 5.3: Test Complete Flow**

```bash
streamlit run app.py

# Test full flow:
# 1. Home page
# 2. Strategy Builder
# 3. Run backtest
# 4. View results
# 5. Explore factors
```

**Task 5.4: Commit Day 5**

```bash
git add .
git commit -m "Day 5: Results dashboard and Factor Explorer complete"
git push
```

**🏁 Day 5 Complete!** Full UI is done!

---

## 📅 Day 6: Polish & Testing

**Goal:** Bug fixes, testing, mobile optimization

**Time Allocation:** 5-6 hours

### Morning (3 hours) - Testing & Bug Fixes

**Task 6.1: Comprehensive Testing**

Create `tests/test_integration.py`:

```python
"""
Integration tests for full backtest flow
"""

import pytest
import pandas as pd
from core.data_loader import DataLoader
from core.factor_engine import FactorEngine
from core.portfolio import PortfolioBuilder
from core.backtester import Backtester, PerformanceAnalytics


def test_full_backtest_flow():
    """Test complete backtest from data to results"""
    
    # 1. Load data
    loader = DataLoader()
    tickers = loader.get_universe('tech')
    data = loader.fetch_data(tickers, '2023-01-01', '2024-01-01')
    
    assert not data.empty
    
    # 2. Calculate factors
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs('Close', level=1, axis=1)
    else:
        prices = data['Close']
    
    factors = FactorEngine.calculate_all_factors(prices, tickers)
    normalized = FactorEngine.normalize_factors(factors)
    
    assert len(factors.columns) == 5
    
    # 3. Build portfolio
    weights = {'momentum': 0.5, 'value': 0.5}
    scores = FactorEngine.combine_factors(normalized, weights)
    
    builder = PortfolioBuilder()
    positions = builder.construct_long_short(scores)
    
    assert len(positions['long']) > 0
    
    # 4. Backtest
    month_ends = prices.resample('M').last().index
    positions_by_date = {date: positions for date in month_ends if date in prices.index}
    
    backtester = Backtester()
    results = backtester.run(prices, positions_by_date)
    
    assert len(results) > 0
    assert results['portfolio_value'].iloc[-1] > 0
    
    # 5. Calculate metrics
    metrics = PerformanceAnalytics.calculate_metrics(results['portfolio_value'])
    
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert metrics['total_return'] != 0


def test_error_handling():
    """Test that errors are handled gracefully"""
    
    loader = DataLoader()
    
    # Test with invalid ticker
    data = loader.fetch_data(['INVALID_TICKER'], '2023-01-01', '2024-01-01')
    # Should return empty DataFrame, not crash
    assert isinstance(data, pd.DataFrame)


def test_performance():
    """Test that operations complete in reasonable time"""
    import time
    
    loader = DataLoader()
    tickers = loader.get_universe('sp500_sample')  # 10 stocks
    
    start_time = time.time()
    data = loader.fetch_data(tickers, '2020-01-01', '2024-01-01')
    data_time = time.time() - start_time
    
    assert data_time < 30, f"Data fetch took {data_time:.1f}s (should be <30s)"
    
    # Test factor calculation speed
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs('Close', level=1, axis=1)
    else:
        prices = data['Close']
    
    start_time = time.time()
    factors = FactorEngine.calculate_all_factors(prices, tickers)
    factor_time = time.time() - start_time
    
    assert factor_time < 5, f"Factor calc took {factor_time:.1f}s (should be <5s)"
```

Run tests:
```bash
pytest tests/ -v --cov=core --cov-report=html

# Check coverage
open htmlcov/index.html
```

**Task 6.2: Mobile Testing**

Test on mobile devices or browser DevTools:

```bash
# Run app
streamlit run app.py

# Test on:
# - iPhone SE (375px width)
# - iPad (768px width)
# - Desktop (1440px width)

# Check:
# ✅ No horizontal scrolling
# ✅ Buttons are tappable (44x44px minimum)
# ✅ Charts render correctly
# ✅ Text is readable
# ✅ Sliders work with touch
```

**Task 6.3: Performance Optimization**

Add caching to expensive operations:

```python
# In pages/1_Strategy_Builder.py

# Add caching decorator
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_process_data(universe_key, start_date, end_date):
    """Cached data loading"""
    loader = DataLoader()
    tickers = loader.get_universe(universe_key)
    data = loader.fetch_data(tickers, start_date, end_date)
    return data, tickers

# Use cached function
data, tickers = load_and_process_data(universe_key, 
                                     start_date.strftime('%Y-%m-%d'),
                                     end_date.strftime('%Y-%m-%d'))
```

### Afternoon (2-3 hours) - UI Polish

**Task 6.4: Add Loading States**

Improve user feedback:

```python
# Better loading indicators
with st.spinner("Running backtest... This may take 30-60 seconds"):
    progress_bar = st.progress(0, text="Starting...")
    
    # Update progress throughout
    progress_bar.progress(25, text="Loading data...")
    # ... load data ...
    
    progress_bar.progress(50, text="Calculating factors...")
    # ... calculate factors ...
    
    progress_bar.progress(75, text="Running simulation...")
    # ... backtest ...
    
    progress_bar.progress(100, text="Complete!")
    progress_bar.empty()
```

**Task 6.5: Error Handling**

Add try-except blocks:

```python
# In Strategy Builder
try:
    data = loader.fetch_data(...)
except Exception as e:
    st.error(f"❌ Failed to load data: {str(e)}")
    st.info("💡 Try selecting a different universe or date range")
    st.stop()
```

**Task 6.6: Add Help Text**

```python
# Add tooltips everywhere
st.slider(
    "Momentum",
    0, 100, 40,
    help="📊 Stocks with strong past performance. Based on 12-month returns."
)

# Add info boxes
with st.expander("ℹ️ What is Sharpe Ratio?"):
    st.write("""
    Sharpe Ratio measures risk-adjusted return. Higher is better.
    
    - < 1.0: Poor
    - 1.0 - 2.0: Good
    - > 2.0: Excellent
    """)
```

**✅ Checkpoint 8:** App is polished and tested

**Task 6.7: Commit Day 6**

```bash
git add .
git commit -m "Day 6: Testing, bug fixes, and polish complete"
git push
```

**🏁 Day 6 Complete!** App is production-ready!

---

## 📅 Day 7: Deployment & Documentation

**Goal:** Deploy to Streamlit Cloud and finalize documentation

**Time Allocation:** 4-5 hours

### Morning (2 hours) - Deployment

**Task 7.1: Prepare for Deployment**

Create `requirements.txt` (production):

```txt
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.0
scipy==1.11.0
yfinance==0.2.28
plotly==5.17.0
matplotlib==3.8.0
python-dateutil==2.8.2
```

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#00D9FF"
backgroundColor = "#F8F9FA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#212121"
font = "sans serif"

[server]
headless = true
enableCORS = false
port = 8501
```

Create `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
venv/

# Data
data/cache/
*.db
*.db-journal

# IDE
.vscode/
.idea/
*.swp

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db
```

**Task 7.2: Deploy to Streamlit Cloud**

```bash
# 1. Commit all changes
git add .
git commit -m "Ready for deployment"
git push

# 2. Go to share.streamlit.io
# 3. Click "New app"
# 4. Connect GitHub repo
# 5. Select:
#    - Repository: yourusername/factor-lab
#    - Branch: main
#    - Main file path: app.py
# 6. Click "Deploy"

# Wait 2-3 minutes...
# Your app will be live at: https://factor-lab-yourusername.streamlit.app
```

**Task 7.3: Test Production App**

```bash
# Open deployed app
# Test all features:
# ✅ Strategy Builder works
# ✅ Backtest runs successfully
# ✅ Results display correctly
# ✅ Charts are interactive
# ✅ Mobile responsive
# ✅ No errors in browser console
```

### Afternoon (2-3 hours) - Documentation

**Task 7.4: Complete README.md**

```markdown
# 📊 Factor Lab

A web-based quantitative investment platform for building, backtesting, and analyzing multi-factor trading strategies.

🔗 **Live Demo:** [factor-lab.streamlit.app](https://factor-lab-yourusername.streamlit.app)

## Features

- ✅ **5 Quantitative Factors:** Momentum, Value, Quality, Size, Low Volatility
- ✅ **Interactive Backtesting:** Simulate strategies on historical data
- ✅ **Performance Analytics:** Sharpe ratio, drawdown, returns distribution
- ✅ **Educational Content:** Learn about each factor with academic references
- ✅ **Mobile Responsive:** Works on phone, tablet, and desktop

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/factor-lab.git
cd factor-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Using the App

1. **Select Universe:** Choose from S&P 500, Tech stocks, or FAANG
2. **Configure Factors:** Adjust weights using sliders
3. **Run Backtest:** Click "Run Backtest" and wait 30-60 seconds
4. **Analyze Results:** View performance metrics and charts
5. **Learn More:** Explore Factor Explorer for educational content

## Technology Stack

- **Frontend:** Streamlit (Python)
- **Data:** yfinance (Yahoo Finance)
- **Storage:** SQLite
- **Visualization:** Plotly
- **Deployment:** Streamlit Cloud

## Project Structure

```
factor-lab/
├── app.py                      # Main entry point
├── pages/
│   ├── 1_Strategy_Builder.py  # Strategy configuration
│   ├── 2_Results.py            # Results dashboard
│   └── 3_Factor_Explorer.py    # Educational content
├── core/
│   ├── data_loader.py          # Data fetching & caching
│   ├── factor_engine.py        # Factor calculations
│   ├── portfolio.py            # Portfolio construction
│   └── backtester.py           # Backtesting engine
├── tests/
│   ├── test_data_loader.py
│   ├── test_factor_engine.py
│   └── test_integration.py
├── data/cache/                 # SQLite cache
├── requirements.txt
└── README.md
```

## Factors Implemented

### 1. Momentum
- **Formula:** 12-month price return (skip last month)
- **Academic Basis:** Jegadeesh & Titman (1993)
- **Typical Return:** 8-12% annually

### 2. Value
- **Formula:** Book-to-Market ratio (inverse P/B)
- **Academic Basis:** Fama & French (1993)
- **Typical Return:** 4-6% annually

### 3. Quality
- **Formula:** Profitability (ROA, profit margins)
- **Academic Basis:** Novy-Marx (2013)
- **Typical Return:** 3-5% annually

### 4. Size
- **Formula:** Log market capitalization
- **Academic Basis:** Banz (1981)
- **Typical Return:** 2-3% annually

### 5. Low Volatility
- **Formula:** 60-day rolling volatility (inverted)
- **Academic Basis:** Ang et al. (2006)
- **Typical Return:** 3-4% annually

## Performance

- **Data Download:** <30 seconds for 100 stocks
- **Factor Calculation:** <5 seconds for 500 stocks
- **Backtest:** <10 seconds for 5-year period
- **Page Load:** <3 seconds on 4G

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov-report=html
open htmlcov/index.html
```

## Development Timeline

- **Day 1:** Project setup & data pipeline
- **Day 2:** Factor engine (5 factors)
- **Day 3:** Portfolio construction & backtesting
- **Day 4:** Streamlit UI - Strategy Builder
- **Day 5:** Results dashboard & Factor Explorer
- **Day 6:** Testing, bug fixes, polish
- **Day 7:** Deployment & documentation

**Total:** 7 days, ~35 hours

## Future Roadmap

### Project 2 (Nov-Dec): Course Project
- [ ] Korean market data (KOSPI, KOSDAQ)
- [ ] Additional factors from research papers
- [ ] Multi-strategy comparison
- [ ] Regime detection

### Project 3 (Jan 2026+): Commercial Product
- [ ] User authentication
- [ ] Save/load strategies
- [ ] Paper trading integration
- [ ] Subscription model ($29/mo)
- [ ] API access for developers

## Academic References

- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers
- Fama, E. F., & French, K. R. (1993). Common risk factors in returns
- Novy-Marx, R. (2013). The other side of value: Gross profitability premium
- Banz, R. W. (1981). The relationship between return and market value
- Ang, A., et al. (2006). The cross-section of volatility and expected returns

## License

MIT License - Feel free to use for learning and research

## Contact

- **Author:** [Your Name]
- **Email:** [your.email@example.com]
- **LinkedIn:** [linkedin.com/in/yourprofile]
- **GitHub:** [github.com/yourusername]

## Acknowledgments

- Data provided by Yahoo Finance (via yfinance)
- Built with Streamlit
- Deployed on Streamlit Cloud

---

**⭐ If you found this useful, please star the repository!**
```

**Task 7.5: Create CHANGELOG.md**

```markdown
# Changelog

All notable changes to Factor Lab will be documented in this file.

## [1.0.0] - 2024-10-31

### Added
- Initial release of Factor Lab
- 5 quantitative factors (Momentum, Value, Quality, Size, Low Volatility)
- Interactive backtesting engine
- Results dashboard with Plotly charts
- Factor Explorer (educational content)
- Mobile-responsive UI
- SQLite caching for faster performance
- Deployed to Streamlit Cloud

### Features
- Multi-factor strategy builder
- Long-short portfolio construction
- Performance analytics (Sharpe, drawdown, etc.)
- Export results (JSON, CSV)
- Pre-defined universes (Tech, S&P 500, FAANG)

### Performance
- Data download: <30s for 100 stocks
- Backtest: <10s for 5-year period
- Page load: <3s

## Upcoming

### [2.0.0] - Q4 2024 (Project 2)
- Korean market support
- Additional factors
- Strategy comparison

### [3.0.0] - Q1 2025 (Project 3)
- User authentication
- Paper trading
- Paid subscriptions
```

**Task 7.6: Create Usage Guide**

Create `USAGE.md`:

```markdown
# Factor Lab - Usage Guide

## Getting Started

### 1. Open the App

Visit: https://factor-lab-yourusername.streamlit.app

### 2. Navigate to Strategy Builder

Click "Build Your Own" or "Try Example Strategy"

### 3. Configure Your Strategy

**Select Universe:**
- Tech (7 stocks): AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA
- S&P 500 Sample (10 stocks): Diversified large-cap
- FAANG: Focus on tech giants

**Adjust Factor Weights:**
- Momentum: 0-100%
- Value: 0-100%
- Quality: 0-100%
- Size: 0-100%
- Low Volatility: 0-100%

Weights will auto-normalize to 100%.

**Portfolio Settings:**
- Strategy Type: Long-Short (default) or Long-Only
- Rebalancing: Monthly, Quarterly, or Annually
- Transaction Cost: Typical is 10 bps (0.1%)

### 4. Run Backtest

Click "🚀 Run Backtest"

Wait 30-60 seconds while the system:
1. Downloads historical data
2. Calculates factor scores
3. Constructs portfolios
4. Simulates trades
5. Analyzes performance

### 5. View Results

**Performance Summary:**
- Total Return
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio (>1 is good)
- Max Drawdown (lowest point)

**Charts:**
- Equity Curve: Portfolio value over time
- Drawdown: Risk visualization
- Returns Heatmap: Monthly performance
- Distribution: Daily returns histogram

**Tabs:**
- Returns: Detailed return analysis
- Drawdown: Risk metrics
- Periodic Returns: Monthly/annual breakdown
- Details: Strategy configuration

### 6. Export Results

Click "📥 Export Results" to download:
- JSON: Full backtest data
- CSV: Equity curve for Excel analysis

## Tips for Best Results

### Strategy Design

**Conservative Strategy:**
```
Value: 40%
Quality: 30%
Low Volatility: 30%
```
- Lower returns but more stable
- Good for risk-averse investors

**Aggressive Strategy:**
```
Momentum: 60%
Value: 40%
```
- Higher potential returns
- More volatile

**Balanced Strategy:**
```
Momentum: 25%
Value: 25%
Quality: 25%
Low Volatility: 25%
```
- Diversified factor exposure
- Moderate risk/return

### Interpreting Metrics

**Sharpe Ratio:**
- < 1.0: Poor risk-adjusted returns
- 1.0 - 2.0: Good
- > 2.0: Excellent (rare!)

**Max Drawdown:**
- < 10%: Very low risk
- 10-20%: Moderate risk
- > 30%: High risk

**CAGR:**
- 5-10%: Conservative
- 10-15%: Moderate
- > 15%: Aggressive (higher risk)

## Troubleshooting

### "No data downloaded"
- Check your internet connection
- Try a different universe
- Reduce date range

### "Backtest failed"
- Ensure factor weights > 0
- Check date range (needs 3+ years)
- Try refreshing the page

### Slow performance
- Use smaller universes (7-10 stocks)
- Reduce date range
- Clear browser cache

## Advanced Usage

### Custom Universes

To add your own stocks, modify `core/data_loader.py`:

```python
universes = {
    'my_custom': ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
}
```

### Adjusting Factor Calculations

Edit `core/factor_engine.py` to customize factor logic:

```python
@staticmethod
def _momentum(prices, lookback=252, skip_last=21):
    # Change lookback period
    lookback = 180  # 9 months instead of 12
    ...
```

## Support

For issues or questions:
- GitHub Issues: [github.com/yourusername/factor-lab/issues]
- Email: your.email@example.com
```

**✅ Checkpoint 9:** Documentation complete

**Task 7.7: Final Testing**

```bash
# Test deployed app
# - Run multiple backtests
# - Test on mobile
# - Test all features
# - Check for bugs

# Update README if needed
git add .
git commit -m "Day 7: Deployment and documentation complete"
git push
```

**🎉 Day 7 Complete! Project 1 is DONE!**

---

## 🎯 Post-Project Checklist

### Immediate (Day 7 Evening)

- [ ] App deployed and accessible
- [ ] README.md complete
- [ ] All tests passing
- [ ] No critical bugs
- [ ] Portfolio piece ready

### Week 2 (Optional Enhancements)

- [ ] Add more predefined universes
- [ ] Improve chart styling
- [ ] Add comparison to S&P 500 benchmark
- [ ] Create tutorial video
- [ ] Share on LinkedIn

### Before Course Starts (Nov 1)

- [ ] Run 10+ backtests to understand factors
- [ ] Read 2-3 academic papers on factor investing
- [ ] Practice explaining your project
- [ ] Prepare questions for professor

---

## 📊 Success Metrics

**Project Completion:**
- ✅ All 5 factors working
- ✅ Backtest produces accurate metrics
- ✅ Mobile-responsive UI
- ✅ Deployed to production
- ✅ Documentation complete

**Learning Outcomes:**
- ✅ Understand factor investing deeply
- ✅ Can explain each factor's logic
- ✅ Know how to construct portfolios
- ✅ Can interpret performance metrics
- ✅ Ready for quant course

**Portfolio Value:**
- ✅ Professional-looking web app
- ✅ Demonstrates full-stack skills
- ✅ Shows quantitative expertise
- ✅ Usable in interviews
- ✅ Foundation for commercial product

---

## 🚀 Next Steps

### For Your Course (Project 2)

1. **Week 1 of course:** Present Project 1 to classmates
2. **Week 2-3:** Add Korean market data
3. **Week 4-6:** Implement factors from course papers
4. **Week 7-8:** Build team presentation

### For Commercial Product (Project 3)

1. **Jan 2026:** Add user authentication
2. **Feb 2026:** Implement paper trading
3. **Mar 2026:** Launch beta with 10 users
4. **Apr 2026:** Pricing model and marketing
5. **May 2026:** Public launch

---

## 💡 Lessons Learned

Document what you learned each day:

**Technical Skills:**
- Data pipeline design
- Factor calculation math
- Backtesting methodology
- Streamlit framework
- Performance optimization

**Project Management:**
- Breaking project into daily chunks
- Managing scope creep
- Prioritizing features
- Testing strategies

**Domain Knowledge:**
- Quantitative factors
- Portfolio construction
- Risk metrics
- Financial data

---

**🎓 You're now ready for your EMBA course!**

**Total Time:** 7 days × 5 hours = 35 hours  
**Lines of Code:** ~2,000  
**Skills Gained:** Quantitative finance, full-stack dev, data engineering  
**Portfolio Value:** High (demonstrates multiple skills)  

**Good luck with your course and your commercial product! 🚀**
