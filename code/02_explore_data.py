"""
Step 2: Exploratory Data Analysis
==================================
Analyze the Korean stock price data to understand basic patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set matplotlib to work without display
import matplotlib
matplotlib.use('Agg')

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load cleaned data
print("\n1. Loading cleaned data...")
stock_prices = pd.read_csv('../data/processed/stock_prices_clean.csv',
                           index_col=0, parse_dates=True)
print(f"   ✓ Loaded: {stock_prices.shape[0]} days × {stock_prices.shape[1]} stocks")

# Calculate daily returns
print("\n2. Calculating daily returns...")
returns = stock_prices.pct_change() * 100  # Convert to percentage
returns = returns.dropna()  # Remove first row (NaN)
print(f"   ✓ Calculated returns: {returns.shape}")

# Basic statistics
print("\n3. Return statistics (all stocks):")
print(f"   Average daily return: {returns.mean().mean():.4f}%")
print(f"   Median daily return: {returns.median().median():.4f}%")
print(f"   Daily volatility (std): {returns.std().mean():.4f}%")
print(f"   Best daily return: {returns.max().max():.2f}%")
print(f"   Worst daily return: {returns.min().min():.2f}%")

# Top performers
print("\n4. Best performing stocks (total return):")
total_returns = (stock_prices.iloc[-1] / stock_prices.iloc[0] - 1) * 100
top_10 = total_returns.nlargest(10)
for i, (stock, ret) in enumerate(top_10.items(), 1):
    print(f"   {i:2d}. {stock:30s} {ret:+8.2f}%")

print("\n5. Worst performing stocks (total return):")
bottom_10 = total_returns.nsmallest(10)
for i, (stock, ret) in enumerate(bottom_10.items(), 1):
    print(f"   {i:2d}. {stock:30s} {ret:+8.2f}%")

# Major stocks analysis
print("\n6. Major Korean stocks performance:")
major_stocks = ['삼성전자', 'SK하이닉스', 'NAVER', '현대차', '기아', 'LG에너지솔루션']
for stock in major_stocks:
    if stock in total_returns.index:
        ret = total_returns[stock]
        vol = returns[stock].std()
        print(f"   {stock:20s} Return: {ret:+7.2f}%  |  Volatility: {vol:.2f}%")

# Market statistics
print("\n7. Market-wide statistics:")
avg_market_return = returns.mean(axis=1)  # Average across all stocks each day
print(f"   Average daily market return: {avg_market_return.mean():.4f}%")
print(f"   Market volatility: {avg_market_return.std():.4f}%")
print(f"   Best market day: {avg_market_return.max():.2f}% on {avg_market_return.idxmax().date()}")
print(f"   Worst market day: {avg_market_return.min():.2f}% on {avg_market_return.idxmin().date()}")

# Positive vs negative days
positive_days = (avg_market_return > 0).sum()
total_days = len(avg_market_return)
print(f"   Positive days: {positive_days}/{total_days} ({positive_days/total_days*100:.1f}%)")

# Create output directory
os.makedirs('../output/figures', exist_ok=True)

# Visualization 1: Market returns over time
print("\n8. Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Average market return over time
ax = axes[0, 0]
cumulative_market = (1 + avg_market_return/100).cumprod()
ax.plot(cumulative_market.index, cumulative_market.values, linewidth=2)
ax.set_title('Korean Market Performance (Equal-Weighted)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Return (1 = start)', fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)

# Plot 2: Distribution of daily returns
ax = axes[0, 1]
ax.hist(avg_market_return.dropna(), bins=50, edgecolor='black', alpha=0.7)
ax.set_title('Distribution of Daily Market Returns', fontsize=12, fontweight='bold')
ax.set_xlabel('Daily Return (%)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.grid(True, alpha=0.3)

# Plot 3: Top 5 stocks vs Market
ax = axes[1, 0]
top_5_stocks = total_returns.nlargest(5).index[:5]
for stock in top_5_stocks:
    if stock in stock_prices.columns:
        normalized = stock_prices[stock] / stock_prices[stock].iloc[0]
        ax.plot(stock_prices.index, normalized, label=stock[:15], linewidth=1.5, alpha=0.7)
ax.plot(cumulative_market.index, cumulative_market.values,
        label='Market Avg', linewidth=2, color='black', linestyle='--')
ax.set_title('Top 5 Performers vs Market', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized Price (1 = start)', fontsize=10)
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# Plot 4: Volatility distribution
ax = axes[1, 1]
stock_volatilities = returns.std()
ax.hist(stock_volatilities, bins=50, edgecolor='black', alpha=0.7)
ax.set_title('Distribution of Stock Volatilities', fontsize=12, fontweight='bold')
ax.set_xlabel('Daily Volatility (Std Dev %)', fontsize=10)
ax.set_ylabel('Number of Stocks', fontsize=10)
ax.axvline(x=stock_volatilities.median(), color='red', linestyle='--',
           linewidth=2, label=f'Median: {stock_volatilities.median():.2f}%')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/figures/01_exploratory_analysis.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: output/figures/01_exploratory_analysis.png")

# Visualization 2: Correlation heatmap for major stocks
print("\n9. Creating correlation analysis...")
fig, ax = plt.subplots(figsize=(10, 8))

# Select major stocks that exist in our data
available_major = [s for s in major_stocks if s in returns.columns]
if len(available_major) >= 3:
    major_returns = returns[available_major[:10]]  # Top 10 or less
    correlation_matrix = major_returns.corr()

    im = ax.imshow(correlation_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(correlation_matrix.index, fontsize=9)

    # Add correlation values
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Correlation Matrix - Major Korean Stocks', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()
    plt.savefig('../output/figures/02_correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: output/figures/02_correlation_matrix.png")

# Save summary statistics to CSV
print("\n10. Saving summary statistics...")
summary_stats = pd.DataFrame({
    'Total_Return_%': total_returns,
    'Avg_Daily_Return_%': returns.mean(),
    'Volatility_%': returns.std(),
    'Max_Daily_Gain_%': returns.max(),
    'Max_Daily_Loss_%': returns.min()
})
summary_stats = summary_stats.sort_values('Total_Return_%', ascending=False)
summary_stats.to_csv('../output/results/stock_summary_stats.csv')
print("   ✓ Saved: output/results/stock_summary_stats.csv")

print("\n" + "="*80)
print("EXPLORATORY ANALYSIS COMPLETE!")
print("="*80)
print("\nKey Findings:")
print(f"  • Market average return: {avg_market_return.mean():.4f}% per day")
print(f"  • Best stock gained: {total_returns.max():.1f}%")
print(f"  • Worst stock lost: {total_returns.min():.1f}%")
print(f"  • Positive market days: {positive_days/total_days*100:.1f}%")
print("\nNext: Run momentum strategy!\n")
