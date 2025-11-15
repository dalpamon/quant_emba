"""
SQL Query Examples for Factor Lab Database (quant1_ tables)
Useful queries for data analysis and debugging
"""

import sqlite3
import pandas as pd


def connect_db(db_path='quant1_data.db'):
    """Create database connection"""
    return sqlite3.connect(db_path)


# ============================================================================
# UNIVERSE QUERIES
# ============================================================================

def query_active_stocks():
    """Get all active stocks in universe"""
    conn = connect_db()
    
    query = """
    SELECT ticker, name, sector, industry, market_cap
    FROM quant1_universe
    WHERE active = 1
    ORDER BY market_cap DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("\n📊 Active Stocks in Universe:")
    print(df)
    return df


def query_stocks_by_sector():
    """Count stocks by sector"""
    conn = connect_db()
    
    query = """
    SELECT 
        sector,
        COUNT(*) as num_stocks,
        AVG(market_cap) as avg_market_cap,
        SUM(market_cap) as total_market_cap
    FROM quant1_universe
    WHERE active = 1
    GROUP BY sector
    ORDER BY num_stocks DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("\n🏢 Stocks by Sector:")
    print(df)
    return df


# ============================================================================
# PRICE QUERIES
# ============================================================================

def query_latest_prices():
    """Get most recent prices for all stocks"""
    conn = connect_db()
    
    query = """
    SELECT 
        p.ticker,
        u.name,
        p.date,
        p.close,
        p.volume
    FROM quant1_prices p
    JOIN quant1_universe u ON p.ticker = u.ticker
    WHERE p.date = (SELECT MAX(date) FROM quant1_prices)
    ORDER BY p.close * p.volume DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("\n💵 Latest Prices:")
    print(df.head(10))
    return df


def query_price_history(ticker, start_date=None, end_date=None):
    """Get price history for a specific stock"""
    conn = connect_db()
    
    query = f"""
    SELECT date, open, high, low, close, volume, adjusted_close
    FROM quant1_prices
    WHERE ticker = '{ticker}'
    """
    
    if start_date:
        query += f" AND date >= '{start_date}'"
    if end_date:
        query += f" AND date <= '{end_date}'"
    
    query += " ORDER BY date"
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\n📈 Price History for {ticker}:")
    print(df)
    return df


# ============================================================================
# FACTOR QUERIES
# ============================================================================

def query_latest_factors():
    """Get factor values for latest date"""
    conn = connect_db()
    
    query = """
    SELECT 
        f.ticker,
        u.name,
        f.momentum_12m,
        f.momentum_6m,
        f.value_pb,
        f.value_pe,
        f.quality_roe,
        f.volatility_60d
    FROM quant1_factors f
    JOIN quant1_universe u ON f.ticker = u.ticker
    WHERE f.date = (SELECT MAX(date) FROM quant1_factors)
    ORDER BY f.momentum_12m DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("\n🎯 Latest Factor Values:")
    print(df)
    return df


def query_top_momentum_stocks(n=10):
    """Find top N momentum stocks"""
    conn = connect_db()
    
    query = f"""
    SELECT 
        f.ticker,
        u.name,
        f.momentum_12m,
        f.momentum_6m,
        f.momentum_3m,
        p.close as current_price
    FROM quant1_factors f
    JOIN quant1_universe u ON f.ticker = u.ticker
    JOIN quant1_prices p ON f.ticker = p.ticker AND f.date = p.date
    WHERE f.date = (SELECT MAX(date) FROM quant1_factors)
    ORDER BY f.momentum_12m DESC
    LIMIT {n}
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\n🚀 Top {n} Momentum Stocks:")
    print(df)
    return df


def query_top_value_stocks(n=10):
    """Find top N value stocks"""
    conn = connect_db()
    
    query = f"""
    SELECT 
        f.ticker,
        u.name,
        f.value_pb,
        f.value_pe,
        fund.pe_ratio,
        fund.pb_ratio
    FROM quant1_factors f
    JOIN quant1_universe u ON f.ticker = u.ticker
    JOIN quant1_fundamentals fund ON f.ticker = fund.ticker
    WHERE f.date = (SELECT MAX(date) FROM quant1_factors)
    AND f.value_pb IS NOT NULL
    ORDER BY f.value_pb DESC
    LIMIT {n}
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\n💎 Top {n} Value Stocks:")
    print(df)
    return df


def query_factor_correlation():
    """Calculate correlation between factors"""
    conn = connect_db()
    
    query = """
    SELECT 
        momentum_12m,
        value_pb,
        quality_roe,
        volatility_60d
    FROM quant1_factors
    WHERE date = (SELECT MAX(date) FROM quant1_factors)
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Calculate correlation matrix
    corr = df.corr()
    
    print("\n🔗 Factor Correlation Matrix:")
    print(corr)
    return corr


# ============================================================================
# FUNDAMENTAL QUERIES
# ============================================================================

def query_fundamentals():
    """Get fundamental metrics for all stocks"""
    conn = connect_db()
    
    query = """
    SELECT 
        f.ticker,
        u.name,
        f.market_cap,
        f.pe_ratio,
        f.pb_ratio,
        f.roe,
        f.profit_margin,
        f.debt_to_equity
    FROM quant1_fundamentals f
    JOIN quant1_universe u ON f.ticker = u.ticker
    ORDER BY f.market_cap DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("\n📊 Fundamental Metrics:")
    print(df)
    return df


def query_quality_stocks(min_roe=0.15, min_margin=0.10):
    """Find high-quality stocks based on profitability"""
    conn = connect_db()
    
    query = f"""
    SELECT 
        f.ticker,
        u.name,
        f.roe,
        f.roa,
        f.profit_margin,
        f.pe_ratio,
        f.pb_ratio
    FROM quant1_fundamentals f
    JOIN quant1_universe u ON f.ticker = u.ticker
    WHERE f.roe >= {min_roe}
    AND f.profit_margin >= {min_margin}
    ORDER BY f.roe DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\n⭐ Quality Stocks (ROE>{min_roe*100}%, Margin>{min_margin*100}%):")
    print(df)
    return df


# ============================================================================
# BACKTEST QUERIES
# ============================================================================

def query_all_backtest_runs():
    """List all backtest runs with performance"""
    conn = connect_db()
    
    query = """
    SELECT 
        br.run_id,
        br.run_name,
        br.start_date,
        br.end_date,
        br.rebalance_freq,
        p.total_return,
        p.cagr,
        p.sharpe_ratio,
        p.max_drawdown,
        br.created_at
    FROM quant1_backtest_runs br
    LEFT JOIN quant1_performance p ON br.run_id = p.run_id
    ORDER BY br.created_at DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("\n🎯 All Backtest Runs:")
    print(df)
    return df


def query_backtest_equity_curve(run_id):
    """Get equity curve for a specific backtest run"""
    conn = connect_db()
    
    query = f"""
    SELECT 
        date,
        portfolio_value,
        daily_return,
        benchmark_return
    FROM quant1_backtest_results
    WHERE run_id = {run_id}
    ORDER BY date
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\n📈 Equity Curve for Run #{run_id}:")
    print(df)
    return df


def query_backtest_positions(run_id, rebalance_date=None):
    """Get positions for a specific backtest run"""
    conn = connect_db()
    
    query = f"""
    SELECT 
        p.rebalance_date,
        p.ticker,
        u.name,
        p.position_type,
        p.weight,
        p.factor_score
    FROM quant1_positions p
    JOIN quant1_universe u ON p.ticker = u.ticker
    WHERE p.run_id = {run_id}
    """
    
    if rebalance_date:
        query += f" AND p.rebalance_date = '{rebalance_date}'"
    
    query += " ORDER BY p.rebalance_date DESC, p.position_type, p.weight DESC"
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\n📋 Positions for Run #{run_id}:")
    print(df)
    return df


def query_best_performing_strategy():
    """Find best performing strategy by Sharpe ratio"""
    conn = connect_db()
    
    query = """
    SELECT 
        br.run_name,
        br.factor_weights,
        br.start_date,
        br.end_date,
        p.total_return,
        p.sharpe_ratio,
        p.max_drawdown,
        p.calmar_ratio
    FROM quant1_backtest_runs br
    JOIN quant1_performance p ON br.run_id = p.run_id
    ORDER BY p.sharpe_ratio DESC
    LIMIT 1
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("\n🏆 Best Performing Strategy:")
    print(df)
    return df


# ============================================================================
# ANALYSIS QUERIES
# ============================================================================

def query_data_coverage():
    """Check data coverage and completeness"""
    conn = connect_db()
    
    query = """
    SELECT 
        'Universe' as table_name,
        COUNT(*) as row_count,
        COUNT(DISTINCT ticker) as unique_tickers
    FROM quant1_universe
    
    UNION ALL
    
    SELECT 
        'Prices' as table_name,
        COUNT(*) as row_count,
        COUNT(DISTINCT ticker) as unique_tickers
    FROM quant1_prices
    
    UNION ALL
    
    SELECT 
        'Factors' as table_name,
        COUNT(*) as row_count,
        COUNT(DISTINCT ticker) as unique_tickers
    FROM quant1_factors
    
    UNION ALL
    
    SELECT 
        'Fundamentals' as table_name,
        COUNT(*) as row_count,
        COUNT(DISTINCT ticker) as unique_tickers
    FROM quant1_fundamentals
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("\n📊 Data Coverage:")
    print(df)
    return df


def query_missing_data():
    """Find stocks with missing price or factor data"""
    conn = connect_db()
    
    query = """
    SELECT 
        u.ticker,
        u.name,
        COUNT(DISTINCT p.date) as price_days,
        COUNT(DISTINCT f.date) as factor_days
    FROM quant1_universe u
    LEFT JOIN quant1_prices p ON u.ticker = p.ticker
    LEFT JOIN quant1_factors f ON u.ticker = f.ticker
    WHERE u.active = 1
    GROUP BY u.ticker, u.name
    HAVING price_days < 1000 OR factor_days < 1000
    ORDER BY price_days
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("\n⚠️  Stocks with Limited Data:")
    print(df)
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run various queries to explore the database
    """
    
    print("="*70)
    print("FACTOR LAB - DATABASE QUERIES")
    print("="*70)
    
    # Check data coverage
    query_data_coverage()
    
    # Universe analysis
    query_stocks_by_sector()
    
    # Factor analysis
    query_top_momentum_stocks(5)
    query_top_value_stocks(5)
    query_quality_stocks()
    
    # Backtest results
    query_all_backtest_runs()
    
    print("\n" + "="*70)
    print("✅ Query examples complete!")
    print("="*70)
