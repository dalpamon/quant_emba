"""
Quick test script to verify data download works
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("\n" + "="*60)
print("Testing Yahoo Finance Data Download")
print("="*60 + "\n")

# Test 1: Check if yfinance works
print("1. Testing yfinance installation...")
try:
    import yfinance
    print(f"   ✓ yfinance version: {yfinance.__version__}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 2: Download sample data
print("\n2. Downloading sample data (AAPL, last 30 days)...")
try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    data = yf.download(
        'AAPL',
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False
    )

    if data.empty:
        print("   ✗ No data downloaded!")
        print("   This could be a firewall/proxy issue")
    else:
        print(f"   ✓ Downloaded {len(data)} days of data")
        print(f"\n   Latest prices:")
        print(data.tail(3))

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Download multiple tickers
print("\n3. Downloading multiple tickers (AAPL, MSFT, GOOGL)...")
try:
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = yf.download(
        tickers,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        group_by='ticker',
        progress=False
    )

    if data.empty:
        print("   ✗ No data downloaded!")
    else:
        print(f"   ✓ Downloaded data for {len(tickers)} tickers")
        print(f"   Shape: {data.shape}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test Complete!")
print("="*60 + "\n")

print("If you see errors above:")
print("  - Check your internet connection")
print("  - Check if a firewall/proxy is blocking Yahoo Finance")
print("  - Try using a VPN if Yahoo Finance is blocked in your region")
print("\nIf all tests passed:")
print("  - Data download is working!")
print("  - The issue is elsewhere in the app")
