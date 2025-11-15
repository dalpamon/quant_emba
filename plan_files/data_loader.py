"""
Data Loader for Factor Lab
Handles data ingestion from yfinance (US) and other sources
"""

import yfinance as yf
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
import json
from typing import List, Dict, Optional


class DataLoader:
    """Load and manage data in quant1_ database"""
    
    def __init__(self, db_path='quant1_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def load_universe(self, tickers: List[str], market='US'):
        """
        Load stock universe into quant1_universe table
        
        Args:
            tickers: List of ticker symbols
            market: Market identifier (US, KR, etc.)
        """
        
        print(f"\n📥 Loading universe: {len(tickers)} stocks")
        print("="*60)
        
        data = []
        failed = []
        
        for i, ticker in enumerate(tickers, 1):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Check if we got valid data
                if not info or 'symbol' not in info:
                    failed.append(ticker)
                    print(f"  ⚠️  [{i}/{len(tickers)}] {ticker}: No data available")
                    continue
                
                data.append({
                    'ticker': ticker,
                    'name': info.get('longName', info.get('shortName', ticker)),
                    'exchange': info.get('exchange', 'UNKNOWN'),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'country': info.get('country', market),
                    'active': 1,
                    'added_date': datetime.now().strftime('%Y-%m-%d'),
                    'delisted_date': None
                })
                
                print(f"  ✅ [{i}/{len(tickers)}] {ticker}: {info.get('longName', ticker)}")
                
            except Exception as e:
                failed.append(ticker)
                print(f"  ❌ [{i}/{len(tickers)}] {ticker}: {str(e)[:50]}")
        
        if data:
            df = pd.DataFrame(data)
            df.to_sql('quant1_universe', self.conn, if_exists='replace', index=False)
            print(f"\n✅ Loaded {len(df)} stocks into quant1_universe")
        
        if failed:
            print(f"⚠️  Failed to load {len(failed)} stocks: {failed}")
            
        return data
        
    def load_prices(self, tickers: List[str], start_date: str, end_date: str):
        """
        Load historical prices into quant1_prices table
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        
        print(f"\n📥 Downloading prices: {start_date} to {end_date}")
        print(f"    Tickers: {len(tickers)}")
        print("="*60)
        
        # Download all at once (much faster)
        try:
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date,
                group_by='ticker',
                auto_adjust=False,  # Keep adjusted close separate
                progress=True,
                threads=True
            )
            
            # Handle single ticker vs multiple
            if len(tickers) == 1:
                # Single ticker returns different structure
                data = {tickers[0]: data}
            
            # Reshape data for database
            rows = []
            success_tickers = []
            
            for ticker in tickers:
                try:
                    if ticker not in data.columns.get_level_values(0):
                        print(f"  ⚠️  {ticker}: No data returned")
                        continue
                        
                    ticker_data = data[ticker] if len(tickers) > 1 else data[ticker]
                    
                    # Skip if no data
                    if ticker_data.empty:
                        print(f"  ⚠️  {ticker}: Empty dataset")
                        continue
                    
                    for date, row in ticker_data.iterrows():
                        # Skip rows with NaN values
                        if pd.isna(row['Close']):
                            continue
                            
                        rows.append({
                            'ticker': ticker,
                            'date': date.strftime('%Y-%m-%d'),
                            'open': float(row['Open']) if not pd.isna(row['Open']) else None,
                            'high': float(row['High']) if not pd.isna(row['High']) else None,
                            'low': float(row['Low']) if not pd.isna(row['Low']) else None,
                            'close': float(row['Close']),
                            'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                            'adjusted_close': float(row['Adj Close']) if not pd.isna(row['Adj Close']) else float(row['Close'])
                        })
                    
                    success_tickers.append(ticker)
                    print(f"  ✅ {ticker}: {len(ticker_data)} days")
                    
                except Exception as e:
                    print(f"  ❌ {ticker}: {str(e)[:50]}")
            
            # Save to database
            if rows:
                df = pd.DataFrame(rows)
                df.to_sql('quant1_prices', self.conn, if_exists='replace', index=False)
                print(f"\n✅ Loaded {len(df):,} price records for {len(success_tickers)} stocks")
            else:
                print("❌ No price data to save")
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            
    def load_fundamentals(self, tickers: List[str], date: Optional[str] = None):
        """
        Load fundamental data into quant1_fundamentals table
        
        Args:
            tickers: List of ticker symbols
            date: Date for fundamentals (defaults to today)
        """
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\n📥 Loading fundamentals for {len(tickers)} stocks")
        print("="*60)
        
        rows = []
        
        for i, ticker in enumerate(tickers, 1):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Check if we got valid data
                if not info or 'symbol' not in info:
                    print(f"  ⚠️  [{i}/{len(tickers)}] {ticker}: No fundamental data")
                    continue
                
                rows.append({
                    'ticker': ticker,
                    'date': date,
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'ps_ratio': info.get('priceToSalesTrailing12Months'),
                    'dividend_yield': info.get('dividendYield'),
                    'roe': info.get('returnOnEquity'),
                    'roa': info.get('returnOnAssets'),
                    'profit_margin': info.get('profitMargins'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'book_value': info.get('bookValue'),
                    'earnings_growth': info.get('earningsGrowth')
                })
                
                print(f"  ✅ [{i}/{len(tickers)}] {ticker}")
                
            except Exception as e:
                print(f"  ❌ [{i}/{len(tickers)}] {ticker}: {str(e)[:50]}")
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_sql('quant1_fundamentals', self.conn, if_exists='append', index=False)
            print(f"\n✅ Loaded fundamentals for {len(df)} stocks")
        else:
            print("❌ No fundamental data to save")
            
    def calculate_and_save_factors(self):
        """
        Calculate factors from prices and fundamentals
        Save to quant1_factors table
        """
        
        print("\n📊 Calculating factors...")
        print("="*60)
        
        # Read prices
        prices_df = pd.read_sql(
            "SELECT ticker, date, adjusted_close FROM quant1_prices ORDER BY ticker, date",
            self.conn
        )
        
        if prices_df.empty:
            print("❌ No price data found in database")
            return
        
        # Pivot to wide format (tickers as columns)
        prices_pivot = prices_df.pivot(index='date', columns='ticker', values='adjusted_close')
        prices_pivot.index = pd.to_datetime(prices_pivot.index)
        
        print(f"  📈 Processing {len(prices_pivot.columns)} stocks")
        print(f"  📅 Date range: {prices_pivot.index[0]} to {prices_pivot.index[-1]}")
        
        # Calculate price-based factors
        factors_list = []
        
        for ticker in prices_pivot.columns:
            price_series = prices_pivot[ticker].dropna()
            
            if len(price_series) < 252:  # Need at least 1 year of data
                print(f"  ⚠️  {ticker}: Insufficient data ({len(price_series)} days)")
                continue
            
            # Calculate returns
            returns = price_series.pct_change()
            
            # Momentum factors
            momentum_12m = price_series.pct_change(252).shift(21)  # 12-mo, skip last month
            momentum_6m = price_series.pct_change(126)
            momentum_3m = price_series.pct_change(63)
            momentum_1m = price_series.pct_change(21)
            
            # Volatility factors
            volatility_60d = returns.rolling(60).std() * np.sqrt(252)  # Annualized
            volatility_20d = returns.rolling(20).std() * np.sqrt(252)
            
            # Volume-based liquidity (would need volume data)
            # For now, we'll add this as None and calculate later if needed
            
            # Combine into records
            for date in price_series.index:
                factors_list.append({
                    'ticker': ticker,
                    'date': date.strftime('%Y-%m-%d'),
                    'momentum_12m': momentum_12m.loc[date] if date in momentum_12m.index else None,
                    'momentum_6m': momentum_6m.loc[date] if date in momentum_6m.index else None,
                    'momentum_3m': momentum_3m.loc[date] if date in momentum_3m.index else None,
                    'momentum_1m': momentum_1m.loc[date] if date in momentum_1m.index else None,
                    'volatility_60d': volatility_60d.loc[date] if date in volatility_60d.index else None,
                    'volatility_20d': volatility_20d.loc[date] if date in volatility_20d.index else None,
                })
        
        factors_df = pd.DataFrame(factors_list)
        
        # Add fundamental factors
        print("  📊 Adding fundamental factors...")
        
        fundamentals_df = pd.read_sql(
            """SELECT ticker, date, pb_ratio, pe_ratio, ps_ratio, 
               market_cap, roe, roa, profit_margin 
               FROM quant1_fundamentals""",
            self.conn
        )
        
        if not fundamentals_df.empty:
            # Take latest fundamental for each ticker
            latest_fundamentals = fundamentals_df.sort_values('date').groupby('ticker').last().reset_index()
            
            # Merge with factors
            factors_df = factors_df.merge(
                latest_fundamentals[['ticker', 'pb_ratio', 'pe_ratio', 'ps_ratio', 
                                   'market_cap', 'roe', 'roa', 'profit_margin']],
                on='ticker',
                how='left'
            )
            
            # Calculate value factors (inverse of P/B, P/E)
            factors_df['value_pb'] = 1 / factors_df['pb_ratio'].replace(0, np.nan)
            factors_df['value_pe'] = 1 / factors_df['pe_ratio'].replace(0, np.nan)
            factors_df['value_ps'] = 1 / factors_df['ps_ratio'].replace(0, np.nan)
            
            # Size factor (log market cap)
            factors_df['size_log_mcap'] = np.log(factors_df['market_cap'].replace(0, np.nan))
            
            # Quality factors
            factors_df['quality_roe'] = factors_df['roe']
            factors_df['quality_roa'] = factors_df['roa']
            factors_df['quality_margin'] = factors_df['profit_margin']
            
            # Liquidity (placeholder for now)
            factors_df['liquidity_volume_20d'] = None
        
        # Select final columns
        final_columns = [
            'ticker', 'date', 'momentum_12m', 'momentum_6m', 'momentum_3m', 'momentum_1m',
            'volatility_60d', 'volatility_20d', 'value_pb', 'value_pe', 'value_ps',
            'size_log_mcap', 'quality_roe', 'quality_roa', 'quality_margin',
            'liquidity_volume_20d'
        ]
        
        factors_df = factors_df[final_columns]
        
        # Save to database
        factors_df.to_sql('quant1_factors', self.conn, if_exists='replace', index=False)
        
        print(f"\n✅ Calculated and saved {len(factors_df):,} factor records")
        print(f"  📊 Unique stocks: {factors_df['ticker'].nunique()}")
        print(f"  📅 Date range: {factors_df['date'].min()} to {factors_df['date'].max()}")
        
    def save_backtest_run(self, run_config: Dict) -> int:
        """
        Save backtest configuration to quant1_backtest_runs
        
        Returns:
            run_id: ID of the saved run
        """
        
        # Convert factor_weights dict to JSON string
        if 'factor_weights' in run_config and isinstance(run_config['factor_weights'], dict):
            run_config['factor_weights'] = json.dumps(run_config['factor_weights'])
        
        df = pd.DataFrame([run_config])
        df.to_sql('quant1_backtest_runs', self.conn, if_exists='append', index=False)
        
        # Get the run_id
        cursor = self.conn.cursor()
        cursor.execute("SELECT run_id FROM quant1_backtest_runs WHERE run_name = ?", 
                      (run_config['run_name'],))
        run_id = cursor.fetchone()[0]
        
        print(f"✅ Saved backtest run: {run_config['run_name']} (ID: {run_id})")
        return run_id
        
    def save_backtest_results(self, run_id: int, equity_curve: pd.Series, 
                             benchmark: Optional[pd.Series] = None):
        """
        Save backtest equity curve to quant1_backtest_results
        
        Args:
            run_id: Backtest run ID
            equity_curve: Series with dates as index and portfolio values
            benchmark: Optional benchmark returns series
        """
        
        results = []
        for i, (date, value) in enumerate(equity_curve.items()):
            daily_return = (value / equity_curve.iloc[i-1] - 1) if i > 0 else 0
            
            benchmark_return = 0
            if benchmark is not None and date in benchmark.index:
                benchmark_return = benchmark.loc[date]
            
            results.append({
                'run_id': run_id,
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': value,
                'daily_return': daily_return,
                'benchmark_return': benchmark_return,
                'active_return': daily_return - benchmark_return,
                'num_long': 0,  # To be filled by portfolio
                'num_short': 0
            })
            
        df = pd.DataFrame(results)
        df.to_sql('quant1_backtest_results', self.conn, if_exists='append', index=False)
        print(f"✅ Saved {len(df)} daily results for run {run_id}")
        
    def save_backtest_positions(self, run_id: int, positions_history: Dict):
        """
        Save position history to quant1_positions
        
        Args:
            run_id: Backtest run ID
            positions_history: Dict of {date: {'long': [...], 'short': [...]}}
        """
        
        rows = []
        for date, positions in positions_history.items():
            # Long positions
            for ticker, pos_data in positions.get('long', {}).items():
                rows.append({
                    'run_id': run_id,
                    'rebalance_date': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'position_type': 'long',
                    'weight': pos_data.get('weight', 0),
                    'shares': pos_data.get('shares', 0),
                    'entry_price': pos_data.get('price', 0),
                    'factor_score': pos_data.get('score', 0)
                })
            
            # Short positions
            for ticker, pos_data in positions.get('short', {}).items():
                rows.append({
                    'run_id': run_id,
                    'rebalance_date': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'position_type': 'short',
                    'weight': pos_data.get('weight', 0),
                    'shares': pos_data.get('shares', 0),
                    'entry_price': pos_data.get('price', 0),
                    'factor_score': pos_data.get('score', 0)
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_sql('quant1_positions', self.conn, if_exists='append', index=False)
            print(f"✅ Saved {len(df)} position records for run {run_id}")
        
    def save_performance_metrics(self, run_id: int, metrics: Dict):
        """
        Save performance summary to quant1_performance
        
        Args:
            run_id: Backtest run ID
            metrics: Dictionary of performance metrics
        """
        
        metrics['run_id'] = run_id
        df = pd.DataFrame([metrics])
        df.to_sql('quant1_performance', self.conn, if_exists='append', index=False)
        print(f"✅ Saved performance metrics for run {run_id}")
        
    def get_factors_for_date(self, date: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get factor data for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            tickers: Optional list of tickers to filter
            
        Returns:
            DataFrame with factor data
        """
        
        query = f"SELECT * FROM quant1_factors WHERE date = '{date}'"
        
        if tickers:
            tickers_str = "', '".join(tickers)
            query += f" AND ticker IN ('{tickers_str}')"
            
        return pd.read_sql(query, self.conn)
        
    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader('quant1_data.db')
    
    # Test with a few stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    print("Testing DataLoader...")
    loader.load_universe(test_tickers)
    loader.load_prices(test_tickers, '2023-01-01', '2024-10-25')
    loader.load_fundamentals(test_tickers)
    loader.calculate_and_save_factors()
    
    loader.close()
    print("\n✅ DataLoader test complete!")
