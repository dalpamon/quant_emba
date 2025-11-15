"""
Database Schema for Factor Lab
All tables use quant1_ prefix for clean namespace management
"""

import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseSchema:
    """Manage database schema creation and initialization"""

    def __init__(self, db_path='quant1_data.db'):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def create_all_tables(self):
        """Create all tables for Factor Lab"""
        logger.info("Creating database schema...")

        self._create_universe_table()
        self._create_prices_table()
        self._create_fundamentals_table()
        self._create_factors_table()
        self._create_backtest_runs_table()
        self._create_backtest_results_table()
        self._create_positions_table()
        self._create_performance_table()

        self.conn.commit()
        logger.info("✅ All tables created successfully")

    def _create_universe_table(self):
        """Stock universe table"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_universe (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                market TEXT DEFAULT 'US',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_universe_sector
            ON quant1_universe(sector)
        ''')

        logger.info("  ✓ quant1_universe table created")

    def _create_prices_table(self):
        """Daily OHLCV price data"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_prices (
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

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
            ON quant1_prices(ticker, date)
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prices_date
            ON quant1_prices(date)
        ''')

        logger.info("  ✓ quant1_prices table created")

    def _create_fundamentals_table(self):
        """Fundamental data (P/E, P/B, ROE, etc.)"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_fundamentals (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                pe_ratio REAL,
                pb_ratio REAL,
                ps_ratio REAL,
                roe REAL,
                roa REAL,
                profit_margin REAL,
                debt_to_equity REAL,
                current_ratio REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_date
            ON quant1_fundamentals(ticker, date)
        ''')

        logger.info("  ✓ quant1_fundamentals table created")

    def _create_factors_table(self):
        """Calculated factor scores"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_factors (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                momentum_12m REAL,
                momentum_6m REAL,
                momentum_3m REAL,
                value_pb REAL,
                value_pe REAL,
                value_ps REAL,
                quality_roe REAL,
                quality_roa REAL,
                quality_margin REAL,
                size_log_mcap REAL,
                volatility_60d REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_factors_date
            ON quant1_factors(date)
        ''')

        logger.info("  ✓ quant1_factors table created")

    def _create_backtest_runs_table(self):
        """Backtest configuration and metadata"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_backtest_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT,
                factor_weights TEXT,
                universe TEXT,
                start_date DATE,
                end_date DATE,
                rebalance_freq TEXT,
                portfolio_type TEXT,
                transaction_cost_bps REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        logger.info("  ✓ quant1_backtest_runs table created")

    def _create_backtest_results_table(self):
        """Daily equity curve for each backtest"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_backtest_results (
                run_id INTEGER,
                date DATE,
                portfolio_value REAL,
                daily_return REAL,
                cumulative_return REAL,
                FOREIGN KEY (run_id) REFERENCES quant1_backtest_runs(run_id),
                PRIMARY KEY (run_id, date)
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_backtest_results_run_id
            ON quant1_backtest_results(run_id)
        ''')

        logger.info("  ✓ quant1_backtest_results table created")

    def _create_positions_table(self):
        """Portfolio positions at each rebalance"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_positions (
                run_id INTEGER,
                rebalance_date DATE,
                ticker TEXT,
                position_type TEXT,
                weight REAL,
                shares INTEGER,
                factor_score REAL,
                FOREIGN KEY (run_id) REFERENCES quant1_backtest_runs(run_id),
                PRIMARY KEY (run_id, rebalance_date, ticker)
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_positions_run_id
            ON quant1_positions(run_id)
        ''')

        logger.info("  ✓ quant1_positions table created")

    def _create_performance_table(self):
        """Summary performance metrics"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS quant1_performance (
                run_id INTEGER PRIMARY KEY,
                total_return REAL,
                cagr REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                calmar_ratio REAL,
                max_drawdown REAL,
                volatility REAL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                num_trades INTEGER,
                FOREIGN KEY (run_id) REFERENCES quant1_backtest_runs(run_id)
            )
        ''')

        logger.info("  ✓ quant1_performance table created")

    def drop_all_tables(self):
        """Drop all quant1_ tables (use with caution!)"""
        logger.warning("⚠️  Dropping all tables...")

        tables = [
            'quant1_performance',
            'quant1_positions',
            'quant1_backtest_results',
            'quant1_backtest_runs',
            'quant1_factors',
            'quant1_fundamentals',
            'quant1_prices',
            'quant1_universe'
        ]

        for table in tables:
            self.cursor.execute(f'DROP TABLE IF EXISTS {table}')

        self.conn.commit()
        logger.info("✅ All tables dropped")

    def get_table_info(self, table_name):
        """Get schema information for a table"""
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        return self.cursor.fetchall()

    def list_all_tables(self):
        """List all quant1_ tables"""
        self.cursor.execute('''
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE 'quant1_%'
            ORDER BY name
        ''')
        return [row[0] for row in self.cursor.fetchall()]

    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Test database creation
    print("\n" + "="*60)
    print("Testing Database Schema Creation")
    print("="*60 + "\n")

    db = DatabaseSchema('quant1_data.db')

    # Create all tables
    db.create_all_tables()

    # List created tables
    print("\nCreated tables:")
    for table in db.list_all_tables():
        print(f"  • {table}")

    # Show schema for one table
    print("\nSample schema (quant1_prices):")
    for col in db.get_table_info('quant1_prices'):
        print(f"  {col}")

    db.close()
    print("\n✅ Database initialization test complete!")
