"""
Database Schema for Factor Lab (Project 1)
All tables use quant1_ prefix for namespace isolation
"""

import sqlite3
import pandas as pd
from datetime import datetime


class QuantDatabase:
    """Database manager for Factor Lab with quant1_ prefix"""
    
    def __init__(self, db_path='quant1_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
    def create_tables(self):
        """Create all tables with quant1_ prefix"""
        
        print("Creating database tables with quant1_ prefix...")
        
        # Table 1: Stock Universe
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_universe (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                country TEXT,
                active BOOLEAN DEFAULT 1,
                added_date TEXT,
                delisted_date TEXT,
                UNIQUE(ticker)
            )
        ''')
        print("  ✅ quant1_universe")
        
        # Table 2: Daily Prices (OHLCV)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adjusted_close REAL,
                FOREIGN KEY (ticker) REFERENCES quant1_universe(ticker),
                UNIQUE(ticker, date)
            )
        ''')
        print("  ✅ quant1_prices")
        
        # Table 3: Fundamentals (Point-in-time)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_fundamentals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                market_cap REAL,
                pe_ratio REAL,
                pb_ratio REAL,
                ps_ratio REAL,
                dividend_yield REAL,
                roe REAL,
                roa REAL,
                profit_margin REAL,
                debt_to_equity REAL,
                current_ratio REAL,
                book_value REAL,
                earnings_growth REAL,
                FOREIGN KEY (ticker) REFERENCES quant1_universe(ticker),
                UNIQUE(ticker, date)
            )
        ''')
        print("  ✅ quant1_fundamentals")
        
        # Table 4: Calculated Factors
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_factors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                momentum_12m REAL,
                momentum_6m REAL,
                momentum_3m REAL,
                momentum_1m REAL,
                volatility_60d REAL,
                volatility_20d REAL,
                value_pb REAL,
                value_pe REAL,
                value_ps REAL,
                size_log_mcap REAL,
                quality_roe REAL,
                quality_roa REAL,
                quality_margin REAL,
                liquidity_volume_20d REAL,
                FOREIGN KEY (ticker) REFERENCES quant1_universe(ticker),
                UNIQUE(ticker, date)
            )
        ''')
        print("  ✅ quant1_factors")
        
        # Table 5: Backtest Runs (Track different strategies)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_backtest_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL UNIQUE,
                strategy_description TEXT,
                factor_weights TEXT,
                universe TEXT,
                start_date TEXT,
                end_date TEXT,
                rebalance_freq TEXT,
                long_pct REAL,
                short_pct REAL,
                initial_capital REAL,
                transaction_cost_bps REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("  ✅ quant1_backtest_runs")
        
        # Table 6: Backtest Results (Daily equity)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                portfolio_value REAL,
                daily_return REAL,
                benchmark_return REAL,
                active_return REAL,
                num_long INTEGER,
                num_short INTEGER,
                FOREIGN KEY (run_id) REFERENCES quant1_backtest_runs(run_id),
                UNIQUE(run_id, date)
            )
        ''')
        print("  ✅ quant1_backtest_results")
        
        # Table 7: Backtest Positions (What stocks held when)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                rebalance_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                position_type TEXT,
                weight REAL,
                shares REAL,
                entry_price REAL,
                factor_score REAL,
                FOREIGN KEY (run_id) REFERENCES quant1_backtest_runs(run_id),
                FOREIGN KEY (ticker) REFERENCES quant1_universe(ticker)
            )
        ''')
        print("  ✅ quant1_positions")
        
        # Table 8: Performance Metrics (Summary stats)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_performance (
                run_id INTEGER PRIMARY KEY,
                total_return REAL,
                cagr REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                volatility REAL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                calmar_ratio REAL,
                information_ratio REAL,
                alpha REAL,
                beta REAL,
                FOREIGN KEY (run_id) REFERENCES quant1_backtest_runs(run_id)
            )
        ''')
        print("  ✅ quant1_performance")
        
        # Create indexes for fast queries
        print("\nCreating indexes...")
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_quant1_prices_ticker_date 
            ON quant1_prices(ticker, date)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_quant1_prices_date 
            ON quant1_prices(date)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_quant1_factors_date 
            ON quant1_factors(date)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_quant1_factors_ticker 
            ON quant1_factors(ticker)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_quant1_positions_run 
            ON quant1_positions(run_id, rebalance_date)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_quant1_results_run 
            ON quant1_backtest_results(run_id)
        ''')
        
        self.conn.commit()
        print("  ✅ All indexes created")
        print("\n✅ Database schema created successfully!")
        
    def get_table_info(self):
        """Display information about all tables"""
        
        tables = [
            'quant1_universe',
            'quant1_prices',
            'quant1_fundamentals',
            'quant1_factors',
            'quant1_backtest_runs',
            'quant1_backtest_results',
            'quant1_positions',
            'quant1_performance'
        ]
        
        print("\n" + "="*60)
        print("DATABASE SUMMARY")
        print("="*60)
        
        for table in tables:
            query = f"SELECT COUNT(*) FROM {table}"
            count = self.cursor.execute(query).fetchone()[0]
            print(f"{table:<30} {count:>10,} rows")
            
        print("="*60)
        
    def vacuum(self):
        """Optimize database (reclaim space, rebuild indexes)"""
        print("Optimizing database...")
        self.cursor.execute("VACUUM")
        print("✅ Database optimized")
        
    def backup(self, backup_path):
        """Create a backup of the database"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        
    def close(self):
        """Close database connection"""
        self.conn.close()
        

if __name__ == "__main__":
    # Test database creation
    db = QuantDatabase('quant1_data.db')
    db.create_tables()
    db.get_table_info()
    db.close()
    print("\n✅ Database ready for use!")
