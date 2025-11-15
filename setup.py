"""
Setup script for Factor Lab
Run this to initialize the database and test data pipeline
"""

import sys
from core import DatabaseSchema, DataLoader

def setup_database():
    """Initialize SQLite database with all tables"""
    print("\n" + "="*60)
    print("Setting up Factor Lab Database")
    print("="*60 + "\n")

    try:
        db = DatabaseSchema('quant1_data.db')
        db.create_all_tables()

        tables = db.list_all_tables()
        print(f"\n✅ Created {len(tables)} tables:")
        for table in tables:
            print(f"  • {table}")

        db.close()

        return True

    except Exception as e:
        print(f"\n❌ Database setup failed: {e}")
        return False


def test_data_loader():
    """Test data loading functionality"""
    print("\n" + "="*60)
    print("Testing Data Loader")
    print("="*60 + "\n")

    try:
        loader = DataLoader('quant1_data.db')

        # Test 1: Show available universes
        print("Available stock universes:")
        for key, name in loader.get_universe_info().items():
            print(f"  • {key}: {name}")

        # Test 2: Load a small universe
        print("\n📥 Loading tech universe...")
        tickers = loader.get_universe_tickers('tech')
        print(f"Tickers: {tickers}")

        data, failed = loader.load_universe(tickers)
        print(f"\n✅ Loaded {len(data)} stocks")
        if failed:
            print(f"⚠️  Failed: {failed}")

        # Test 3: Fetch sample price data (1 month)
        print("\n📊 Fetching sample price data (Jan 2024)...")
        prices = loader.fetch_prices(
            tickers[:3],  # Just first 3 stocks for quick test
            '2024-01-01',
            '2024-01-31'
        )

        if not prices.empty:
            print(f"✅ Downloaded {len(prices)} days of data")
        else:
            print("⚠️  No data downloaded")

        loader.close()

        return True

    except Exception as e:
        print(f"\n❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run full setup"""
    print("\n🚀 Factor Lab Setup\n")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False

    print(f"✓ Python version: {sys.version.split()[0]}")

    # Setup database
    if not setup_database():
        return False

    # Test data loader
    print("\n" + "="*60)
    response = input("\nTest data loader? This will download sample data. (y/n): ")

    if response.lower() == 'y':
        if not test_data_loader():
            print("\n⚠️  Data loader test failed, but database is set up.")
            print("You can still run the app, but data downloads may not work.")
    else:
        print("\nSkipping data loader test.")

    # Done
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run the app: streamlit run app.py")
    print("\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
