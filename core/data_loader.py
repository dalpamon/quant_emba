"""
Data Loader for Factor Lab
Handles data ingestion from yfinance with caching
"""

import yfinance as yf
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import json
from typing import List, Dict, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage stock data with caching"""

    # Predefined universes
    UNIVERSES = {
        'tech': {
            'name': 'Tech Giants (7 stocks)',
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
        },
        'sp500_sample': {
            'name': 'S&P 500 Sample (10 stocks)',
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'V', 'WMT', 'PG', 'JNJ', 'UNH']
        },
        'faang': {
            'name': 'FAANG',
            'tickers': ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']
        },
        'dow_sample': {
            'name': 'Dow Jones Sample',
            'tickers': ['AAPL', 'MSFT', 'JPM', 'V', 'HD', 'DIS', 'BA', 'NKE', 'MCD', 'INTC']
        }
    }

    def __init__(self, db_path='quant1_data.db'):
        """
        Initialize data loader

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)

    def get_universe_info(self):
        """Get information about all predefined universes"""
        return {k: v['name'] for k, v in self.UNIVERSES.items()}

    def get_universe_tickers(self, universe_key: str) -> List[str]:
        """
        Get tickers for a predefined universe

        Args:
            universe_key: Key from UNIVERSES dict

        Returns:
            List of ticker symbols
        """
        if universe_key not in self.UNIVERSES:
            raise ValueError(f"Unknown universe: {universe_key}. "
                           f"Available: {list(self.UNIVERSES.keys())}")

        return self.UNIVERSES[universe_key]['tickers']

    def load_universe(self, tickers: List[str]):
        """
        Load stock universe info into database

        Args:
            tickers: List of ticker symbols
        """
        logger.info(f"\n📥 Loading universe: {len(tickers)} stocks")

        data = []
        failed = []

        for i, ticker in enumerate(tickers, 1):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Check if we got valid data
                if not info or 'symbol' not in info:
                    failed.append(ticker)
                    logger.warning(f"  ⚠️  [{i}/{len(tickers)}] {ticker}: No data available")
                    continue

                data.append({
                    'ticker': ticker,
                    'name': info.get('longName', info.get('shortName', ticker)),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', None),
                    'market': 'US'
                })

                logger.info(f"  ✓ [{i}/{len(tickers)}] {ticker}: {data[-1]['name']}")

            except Exception as e:
                failed.append(ticker)
                logger.error(f"  ❌ [{i}/{len(tickers)}] {ticker}: {str(e)}")

        if data:
            df = pd.DataFrame(data)
            df.to_sql('quant1_universe', self.conn, if_exists='replace', index=False)
            logger.info(f"\n✅ Loaded {len(data)} stocks into database")

        if failed:
            logger.warning(f"⚠️  Failed to load {len(failed)} stocks: {failed}")

        return data, failed

    def fetch_prices(self, tickers: List[str], start_date: str, end_date: str,
                    force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch OHLCV price data with caching

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with MultiIndex columns (ticker, field)
        """
        if not force_refresh:
            # Try to load from cache
            cached = self._load_prices_from_cache(tickers, start_date, end_date)
            if cached is not None and not cached.empty:
                logger.info(f"✅ Loaded {len(tickers)} tickers from cache")
                return cached

        # Download from yfinance
        logger.info(f"\n📥 Downloading {len(tickers)} tickers from {start_date} to {end_date}...")

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
            self._save_prices_to_cache(data, tickers)
            logger.info(f"✅ Downloaded and cached {len(tickers)} tickers")

            return data

        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            # Try to return cached data even if expired
            cached = self._load_prices_from_cache(tickers, start_date, end_date)
            if cached is not None:
                logger.warning("⚠️  Using cached data due to download failure")
                return cached
            return pd.DataFrame()

    def _load_prices_from_cache(self, tickers: List[str], start_date: str,
                                end_date: str) -> Optional[pd.DataFrame]:
        """Load price data from cache"""
        try:
            placeholders = ','.join(['?'] * len(tickers))
            query = f"""
                SELECT ticker, date, open, high, low, close, volume, adjusted_close
                FROM quant1_prices
                WHERE ticker IN ({placeholders})
                AND date BETWEEN ? AND ?
                ORDER BY date, ticker
            """

            df = pd.read_sql_query(
                query,
                self.conn,
                params=tuple(tickers) + (start_date, end_date),
                parse_dates=['date'],
                index_col='date'
            )

            if df.empty:
                return None

            # Reshape to yfinance format (MultiIndex columns)
            result = {}
            for ticker in tickers:
                ticker_data = df[df['ticker'] == ticker].drop(columns=['ticker'])
                if not ticker_data.empty:
                    result[ticker] = ticker_data.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                        'adjusted_close': 'Adj Close'
                    })

            if not result:
                return None

            # Create MultiIndex DataFrame
            final_df = pd.concat(result, axis=1)
            return final_df

        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None

    def _save_prices_to_cache(self, data: pd.DataFrame, tickers: List[str]):
        """Save price data to cache"""
        try:
            records = []

            # Handle both single and multiple ticker cases
            if isinstance(data.columns, pd.MultiIndex):
                # Multiple tickers
                for ticker in tickers:
                    if ticker not in data.columns.get_level_values(0):
                        continue

                    ticker_data = data[ticker]
                    for date, row in ticker_data.iterrows():
                        records.append({
                            'ticker': ticker,
                            'date': date.strftime('%Y-%m-%d'),
                            'open': row.get('Open', None),
                            'high': row.get('High', None),
                            'low': row.get('Low', None),
                            'close': row.get('Close', None),
                            'volume': int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None,
                            'adjusted_close': row.get('Close', None)
                        })
            else:
                # Single ticker
                ticker = tickers[0]
                for date, row in data.iterrows():
                    records.append({
                        'ticker': ticker,
                        'date': date.strftime('%Y-%m-%d'),
                        'open': row.get('Open', None),
                        'high': row.get('High', None),
                        'low': row.get('Low', None),
                        'close': row.get('Close', None),
                        'volume': int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None,
                        'adjusted_close': row.get('Close', None)
                    })

            if records:
                # Delete existing records for these tickers/dates
                for ticker in tickers:
                    self.conn.execute(
                        "DELETE FROM quant1_prices WHERE ticker = ?",
                        (ticker,)
                    )

                # Insert new records
                df = pd.DataFrame(records)
                df.to_sql('quant1_prices', self.conn, if_exists='append', index=False)
                self.conn.commit()

                logger.info(f"  💾 Cached {len(records)} price records")

        except Exception as e:
            logger.error(f"Cache save failed: {e}")

    def fetch_fundamentals(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch fundamental data for tickers

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with fundamental metrics
        """
        logger.info(f"\n📊 Fetching fundamentals for {len(tickers)} tickers...")

        fundamentals = []

        for i, ticker in enumerate(tickers, 1):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                fundamentals.append({
                    'ticker': ticker,
                    'date': datetime.now().date().strftime('%Y-%m-%d'),
                    'pe_ratio': info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'ps_ratio': info.get('priceToSalesTrailing12Months'),
                    'roe': info.get('returnOnEquity'),
                    'roa': info.get('returnOnAssets'),
                    'profit_margin': info.get('profitMargins'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio')
                })

                logger.info(f"  ✓ [{i}/{len(tickers)}] {ticker}")

            except Exception as e:
                logger.error(f"  ❌ [{i}/{len(tickers)}] {ticker}: {str(e)}")
                fundamentals.append({
                    'ticker': ticker,
                    'date': datetime.now().date().strftime('%Y-%m-%d'),
                    'pe_ratio': None, 'pb_ratio': None, 'ps_ratio': None,
                    'roe': None, 'roa': None, 'profit_margin': None,
                    'debt_to_equity': None, 'current_ratio': None
                })

        df = pd.DataFrame(fundamentals)

        # Save to database
        if not df.empty:
            # Delete existing fundamentals for these tickers
            for ticker in tickers:
                self.conn.execute(
                    "DELETE FROM quant1_fundamentals WHERE ticker = ?",
                    (ticker,)
                )

            df.to_sql('quant1_fundamentals', self.conn, if_exists='append', index=False)
            self.conn.commit()
            logger.info(f"✅ Saved fundamentals to database")

        return df.set_index('ticker')

    def get_cached_data_info(self, ticker: str) -> Dict:
        """Get information about cached data for a ticker"""
        query = """
            SELECT
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(*) as num_records
            FROM quant1_prices
            WHERE ticker = ?
        """

        result = pd.read_sql_query(query, self.conn, params=(ticker,))

        if result.empty or pd.isna(result.iloc[0]['earliest_date']):
            return {'has_data': False}

        return {
            'has_data': True,
            'earliest_date': result.iloc[0]['earliest_date'],
            'latest_date': result.iloc[0]['latest_date'],
            'num_records': int(result.iloc[0]['num_records'])
        }

    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Test data loader
    print("\n" + "="*60)
    print("Testing Data Loader")
    print("="*60 + "\n")

    loader = DataLoader('quant1_data.db')

    # Test 1: List universes
    print("Available universes:")
    for key, name in loader.get_universe_info().items():
        print(f"  {key}: {name}")

    # Test 2: Load universe
    print("\nLoading tech universe...")
    tickers = loader.get_universe_tickers('tech')
    data, failed = loader.load_universe(tickers)

    # Test 3: Fetch prices (small date range for quick test)
    print("\nFetching price data...")
    prices = loader.fetch_prices(tickers[:3], '2024-01-01', '2024-01-31')
    print(f"Downloaded {len(prices)} days of data")
    print(f"Columns: {prices.columns.tolist()[:5]}...")

    # Test 4: Test cache
    print("\nTesting cache...")
    prices_cached = loader.fetch_prices(tickers[:3], '2024-01-01', '2024-01-31')
    print(f"Loaded from cache: {len(prices_cached)} days")

    # Test 5: Fetch fundamentals
    print("\nFetching fundamentals...")
    fundamentals = loader.fetch_fundamentals(tickers[:3])
    print(f"Fundamentals shape: {fundamentals.shape}")
    print(fundamentals[['pe_ratio', 'pb_ratio', 'roe']].head())

    loader.close()
    print("\n✅ Data loader test complete!")
