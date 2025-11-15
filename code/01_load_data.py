"""
Step 1: Load and Clean FnGuide Data
====================================
This script loads the Korean stock price data from FnGuide Excel file
and prepares it for analysis.
"""

import pandas as pd
import numpy as np
import os

print("="*80)
print("LOADING FNGUIDE DATA")
print("="*80)

# File path
data_file = '../data/quant.xlsx'

# Read the Excel file
print("\n1. Reading Excel file...")
df = pd.read_excel(data_file, sheet_name='Sheet1 (2)', header=None)
print(f"   ✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Extract company names from row 10 (index 9)
print("\n2. Extracting company names...")
companies = df.iloc[9, 1:].dropna().tolist()
print(f"   ✓ Found {len(companies)} companies")
print(f"   First 5: {companies[:5]}")

# Extract dates from column A, starting from row 15 (index 14)
print("\n3. Extracting dates...")
dates = pd.to_datetime(df.iloc[14:, 0].values)
print(f"   ✓ Found {len(dates)} trading days")
print(f"   Date range: {dates[0].date()} to {dates[-1].date()}")

# Extract price data (rows 15+, columns 1+)
print("\n4. Extracting price data...")
prices = df.iloc[14:, 1:len(companies)+1].values

# Create DataFrame with proper labels
stock_prices = pd.DataFrame(
    prices,
    index=dates,
    columns=companies
)

print(f"   ✓ Created price DataFrame: {stock_prices.shape}")
print(f"   Memory usage: {stock_prices.memory_usage().sum() / 1024**2:.2f} MB")

# Data quality check
print("\n5. Data quality check...")
total_values = stock_prices.size
missing_values = stock_prices.isna().sum().sum()
print(f"   Total data points: {total_values:,}")
print(f"   Missing values: {missing_values:,} ({missing_values/total_values*100:.2f}%)")

# Check which stocks have too much missing data
missing_pct = stock_prices.isna().sum() / len(stock_prices) * 100
stocks_with_issues = missing_pct[missing_pct > 20]
print(f"   Stocks with >20% missing data: {len(stocks_with_issues)}")

# Remove stocks with excessive missing data
print("\n6. Cleaning data...")
good_stocks = missing_pct[missing_pct <= 20].index
stock_prices_clean = stock_prices[good_stocks].copy()
print(f"   ✓ Kept {len(good_stocks)} stocks (removed {len(companies) - len(good_stocks)})")

# Forward fill remaining missing values
stock_prices_clean = stock_prices_clean.fillna(method='ffill')
# Backward fill any remaining NAs at the start
stock_prices_clean = stock_prices_clean.fillna(method='bfill')

remaining_missing = stock_prices_clean.isna().sum().sum()
print(f"   ✓ After forward/backward fill: {remaining_missing} missing values")

# Save cleaned data
print("\n7. Saving cleaned data...")
os.makedirs('../data/processed', exist_ok=True)
stock_prices_clean.to_csv('../data/processed/stock_prices_clean.csv')
print(f"   ✓ Saved to: data/processed/stock_prices_clean.csv")

# Display sample
print("\n8. Sample of cleaned data:")
print(stock_prices_clean.head())

print("\n" + "="*80)
print("DATA LOADING COMPLETE!")
print("="*80)
print(f"\nFinal dataset: {stock_prices_clean.shape[0]} days × {stock_prices_clean.shape[1]} stocks")
print(f"Ready for analysis!\n")
