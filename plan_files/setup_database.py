"""
Setup Script for Factor Lab Database
Initializes database with sample data
"""

from database_schema import QuantDatabase
from data_loader import DataLoader
from datetime import datetime
import sys


def setup_quant1_database(tickers=None, start_date='2020-01-01', end_date='2024-10-25'):
    """
    Complete setup for Factor Lab database
    
    Args:
        tickers: List of ticker symbols (defaults to tech stocks)
        start_date: Historical data start date
        end_date: Historical data end date
    """
    
    if tickers is None:
        # Default universe: Top tech stocks
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
            'ORCL', 'ADBE', 'CSCO', 'AVGO', 'QCOM',
            'TXN', 'AMAT', 'MU', 'LRCX', 'KLAC'
        ]
    
    print("=" * 70)
    print("FACTOR LAB - DATABASE SETUP")
    print("=" * 70)
    print(f"\n📦 Setting up database with:")
    print(f"  • {len(tickers)} stocks")
    print(f"  • Date range: {start_date} to {end_date}")
    print(f"  • Table prefix: quant1_")
    print("\n" + "=" * 70)
    
    try:
        # Step 1: Create database and tables
        print("\n🔨 STEP 1: Creating database schema...")
        db = QuantDatabase('quant1_data.db')
        db.create_tables()
        db.close()
        
        # Step 2: Load stock universe
        print("\n🔨 STEP 2: Loading stock universe...")
        loader = DataLoader('quant1_data.db')
        loader.load_universe(tickers, market='US')
        
        # Step 3: Load historical prices
        print("\n🔨 STEP 3: Downloading historical prices...")
        print("   (This may take a few minutes...)")
        loader.load_prices(tickers, start_date, end_date)
        
        # Step 4: Load fundamentals
        print("\n🔨 STEP 4: Loading fundamental data...")
        loader.load_fundamentals(tickers)
        
        # Step 5: Calculate factors
        print("\n🔨 STEP 5: Calculating factors...")
        loader.calculate_and_save_factors()
        
        loader.close()
        
        # Verify setup
        print("\n🔍 STEP 6: Verifying setup...")
        db = QuantDatabase('quant1_data.db')
        db.get_table_info()
        db.close()
        
        print("\n" + "=" * 70)
        print("✅ DATABASE SETUP COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Database file: quant1_data.db")
        print(f"📈 Ready to backtest!")
        print(f"\nNext steps:")
        print(f"  1. Run: python app.py")
        print(f"  2. Open: http://localhost:8501")
        print(f"  3. Start backtesting!")
        print("\n" + "=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Setup failed")
        print(f"   {str(e)}")
        print(f"\nPlease check your internet connection and try again.")
        return False


def add_custom_universe(universe_name, tickers, start_date='2020-01-01', end_date='2024-10-25'):
    """
    Add a custom stock universe to existing database
    
    Args:
        universe_name: Name for this universe (e.g., 'energy', 'finance')
        tickers: List of ticker symbols
        start_date: Historical data start date
        end_date: Historical data end date
    """
    
    print(f"\n📦 Adding custom universe: {universe_name}")
    print(f"   Tickers: {len(tickers)}")
    
    loader = DataLoader('quant1_data.db')
    
    # Load universe
    loader.load_universe(tickers, market='US')
    
    # Load prices
    loader.load_prices(tickers, start_date, end_date)
    
    # Load fundamentals
    loader.load_fundamentals(tickers)
    
    # Recalculate factors
    loader.calculate_and_save_factors()
    
    loader.close()
    
    print(f"✅ Added {universe_name} universe successfully")


def quick_setup():
    """Quick setup with minimal stocks for testing"""
    
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("\n⚡ QUICK SETUP MODE")
    print("   Using 5 stocks for fast testing\n")
    
    return setup_quant1_database(
        tickers=test_tickers,
        start_date='2023-01-01',
        end_date='2024-10-25'
    )


if __name__ == "__main__":
    """
    Run setup from command line
    
    Usage:
        python setup_database.py              # Full setup
        python setup_database.py --quick      # Quick setup for testing
        python setup_database.py --custom     # Custom ticker list
    """
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            success = quick_setup()
            
        elif sys.argv[1] == '--custom':
            print("\n📝 Enter tickers (comma-separated):")
            ticker_input = input("> ")
            tickers = [t.strip().upper() for t in ticker_input.split(',')]
            
            print("\n📅 Start date (YYYY-MM-DD) [default: 2020-01-01]:")
            start_input = input("> ").strip()
            start_date = start_input if start_input else '2020-01-01'
            
            print("\n📅 End date (YYYY-MM-DD) [default: 2024-10-25]:")
            end_input = input("> ").strip()
            end_date = end_input if end_input else '2024-10-25'
            
            success = setup_quant1_database(tickers, start_date, end_date)
            
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python setup_database.py [--quick | --custom]")
            success = False
    else:
        # Default: full setup
        success = setup_quant1_database()
    
    sys.exit(0 if success else 1)
