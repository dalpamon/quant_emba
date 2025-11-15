"""
Step 3: Momentum Strategy Implementation
=========================================
Implement and backtest a momentum trading strategy on Korean stocks.

Strategy:
- Every day, rank stocks by past 20-day return
- Buy top 10% performers (Winners)
- Sell bottom 10% performers (Losers)
- Hold for 1 day, then rebalance
- Measure: Do winners beat losers?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats

print("="*80)
print("MOMENTUM STRATEGY BACKTEST")
print("="*80)

# Load cleaned data
print("\n1. Loading data...")
stock_prices = pd.read_csv('../data/processed/stock_prices_clean.csv',
                           index_col=0, parse_dates=True)
print(f"   ✓ Loaded: {stock_prices.shape[0]} days × {stock_prices.shape[1]} stocks")

# Calculate daily returns
print("\n2. Calculating returns...")
returns = stock_prices.pct_change() * 100  # Daily returns in %
returns = returns.dropna()
print(f"   ✓ Returns shape: {returns.shape}")

# Strategy parameters
LOOKBACK_PERIOD = 20  # Use past 20 days to calculate momentum
TOP_PCT = 10          # Top 10% = winners
BOTTOM_PCT = 10       # Bottom 10% = losers

print(f"\n3. Strategy Parameters:")
print(f"   Lookback period: {LOOKBACK_PERIOD} days")
print(f"   Winner portfolio: Top {TOP_PCT}%")
print(f"   Loser portfolio: Bottom {BOTTOM_PCT}%")

# Calculate momentum signal (past N-day return)
print(f"\n4. Calculating {LOOKBACK_PERIOD}-day momentum...")
momentum = stock_prices.pct_change(LOOKBACK_PERIOD) * 100
print(f"   ✓ Momentum calculated")

# Initialize results storage
portfolio_returns = []
dates_list = []

# Backtest loop
print(f"\n5. Running backtest...")
valid_dates = momentum.index[LOOKBACK_PERIOD+1:]  # Skip initial period
total_days = len(valid_dates)

for i, date in enumerate(valid_dates[:-1], 1):  # Stop before last day
    # Get momentum values for this date
    mom_today = momentum.loc[date]

    # Remove NaN values
    mom_today = mom_today.dropna()

    if len(mom_today) < 20:  # Need minimum stocks
        continue

    # Rank stocks by momentum
    ranks = mom_today.rank(pct=True) * 100  # Convert to percentile (0-100)

    # Identify winners and losers
    winners = ranks[ranks >= (100 - TOP_PCT)].index
    losers = ranks[ranks <= BOTTOM_PCT].index

    # Get next day's return
    next_date = valid_dates[valid_dates.get_loc(date) + 1]
    next_returns = returns.loc[next_date]

    # Calculate portfolio returns (equal-weighted)
    winner_return = next_returns[winners].mean()
    loser_return = next_returns[losers].mean()

    # Store results
    portfolio_returns.append({
        'date': next_date,
        'winner_return': winner_return,
        'loser_return': loser_return,
        'long_short': winner_return - loser_return,
        'n_winners': len(winners),
        'n_losers': len(losers)
    })

    if i % 50 == 0:
        print(f"   Processed {i}/{total_days} days...")

print(f"   ✓ Backtest complete: {len(portfolio_returns)} trading days")

# Convert to DataFrame
results_df = pd.DataFrame(portfolio_returns)
results_df.set_index('date', inplace=True)

# Calculate statistics
print("\n6. Performance Statistics:")
print("="*80)

# Winner portfolio
winner_avg = results_df['winner_return'].mean()
winner_std = results_df['winner_return'].std()
winner_sharpe = winner_avg / winner_std * np.sqrt(252)  # Annualized Sharpe
winner_win_rate = (results_df['winner_return'] > 0).sum() / len(results_df) * 100

print("\nWINNER PORTFOLIO (Top 10%):")
print(f"  Average daily return:     {winner_avg:+.4f}%")
print(f"  Annualized return:        {(1 + winner_avg/100)**252 - 1:+.2%}")
print(f"  Daily volatility:         {winner_std:.4f}%")
print(f"  Sharpe ratio (annual):    {winner_sharpe:.2f}")
print(f"  Win rate:                 {winner_win_rate:.1f}%")

# Loser portfolio
loser_avg = results_df['loser_return'].mean()
loser_std = results_df['loser_return'].std()
loser_sharpe = loser_avg / loser_std * np.sqrt(252)
loser_win_rate = (results_df['loser_return'] > 0).sum() / len(results_df) * 100

print("\nLOSER PORTFOLIO (Bottom 10%):")
print(f"  Average daily return:     {loser_avg:+.4f}%")
print(f"  Annualized return:        {(1 + loser_avg/100)**252 - 1:+.2%}")
print(f"  Daily volatility:         {loser_std:.4f}%")
print(f"  Sharpe ratio (annual):    {loser_sharpe:.2f}")
print(f"  Win rate:                 {loser_win_rate:.1f}%")

# Long-short strategy
ls_avg = results_df['long_short'].mean()
ls_std = results_df['long_short'].std()
ls_sharpe = ls_avg / ls_std * np.sqrt(252)
ls_win_rate = (results_df['long_short'] > 0).sum() / len(results_df) * 100

print("\nLONG-SHORT STRATEGY (Winners - Losers):")
print(f"  Average daily return:     {ls_avg:+.4f}%")
print(f"  Annualized return:        {(1 + ls_avg/100)**252 - 1:+.2%}")
print(f"  Daily volatility:         {ls_std:.4f}%")
print(f"  Sharpe ratio (annual):    {ls_sharpe:.2f}")
print(f"  Win rate:                 {ls_win_rate:.1f}%")

# Statistical significance test
t_stat, p_value = stats.ttest_1samp(results_df['long_short'], 0)

print("\nSTATISTICAL SIGNIFICANCE:")
print(f"  T-statistic:              {t_stat:.4f}")
print(f"  P-value:                  {p_value:.6f}")
if p_value < 0.01:
    print(f"  Conclusion:               *** HIGHLY SIGNIFICANT (p < 0.01) ***")
elif p_value < 0.05:
    print(f"  Conclusion:               ** SIGNIFICANT (p < 0.05) **")
else:
    print(f"  Conclusion:               Not significant (p >= 0.05)")

print("="*80)

# Calculate cumulative returns
print("\n7. Calculating cumulative returns...")
results_df['winner_cumulative'] = (1 + results_df['winner_return']/100).cumprod()
results_df['loser_cumulative'] = (1 + results_df['loser_return']/100).cumprod()
results_df['long_short_cumulative'] = (1 + results_df['long_short']/100).cumprod()

# Save results
print("\n8. Saving results...")
results_df.to_csv('../output/results/momentum_backtest_results.csv')
print("   ✓ Saved: output/results/momentum_backtest_results.csv")

# Create summary statistics
summary = pd.DataFrame({
    'Portfolio': ['Winners (Top 10%)', 'Losers (Bottom 10%)', 'Long-Short'],
    'Avg_Daily_Return_%': [winner_avg, loser_avg, ls_avg],
    'Annual_Return_%': [
        ((1 + winner_avg/100)**252 - 1) * 100,
        ((1 + loser_avg/100)**252 - 1) * 100,
        ((1 + ls_avg/100)**252 - 1) * 100
    ],
    'Volatility_%': [winner_std, loser_std, ls_std],
    'Sharpe_Ratio': [winner_sharpe, loser_sharpe, ls_sharpe],
    'Win_Rate_%': [winner_win_rate, loser_win_rate, ls_win_rate]
})
summary.to_csv('../output/results/momentum_summary.csv', index=False)
print("   ✓ Saved: output/results/momentum_summary.csv")

# Visualizations
print("\n9. Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Cumulative returns
ax = axes[0, 0]
ax.plot(results_df.index, results_df['winner_cumulative'],
        label=f'Winners (Top {TOP_PCT}%)', linewidth=2, color='green')
ax.plot(results_df.index, results_df['loser_cumulative'],
        label=f'Losers (Bottom {BOTTOM_PCT}%)', linewidth=2, color='red')
ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_title(f'Momentum Strategy - Cumulative Returns ({LOOKBACK_PERIOD}-day lookback)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Return (1 = start)', fontsize=11)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Plot 2: Long-Short cumulative return
ax = axes[0, 1]
ax.plot(results_df.index, results_df['long_short_cumulative'],
        linewidth=2, color='blue')
ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_title(f'Long-Short Strategy (Winners - Losers)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Return (1 = start)', fontsize=11)
ax.grid(True, alpha=0.3)
final_return = (results_df['long_short_cumulative'].iloc[-1] - 1) * 100
ax.text(0.02, 0.98, f'Total Return: {final_return:+.1f}%',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Distribution of daily long-short returns
ax = axes[1, 0]
ax.hist(results_df['long_short'], bins=50, edgecolor='black', alpha=0.7, color='purple')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax.axvline(x=ls_avg, color='green', linestyle='--', linewidth=2,
           label=f'Mean: {ls_avg:.3f}%')
ax.set_title('Distribution of Daily Long-Short Returns',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Daily Return (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Rolling 30-day average return
ax = axes[1, 1]
rolling_30d = results_df['long_short'].rolling(30).mean()
ax.plot(rolling_30d.index, rolling_30d, linewidth=2, color='orange')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=ls_avg, color='green', linestyle='--', alpha=0.7,
           linewidth=1.5, label=f'Overall Avg: {ls_avg:.3f}%')
ax.set_title('Rolling 30-Day Average Return (Long-Short)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('30-Day Avg Return (%)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.fill_between(rolling_30d.index, 0, rolling_30d,
                 where=(rolling_30d >= 0), alpha=0.3, color='green', interpolate=True)
ax.fill_between(rolling_30d.index, 0, rolling_30d,
                 where=(rolling_30d < 0), alpha=0.3, color='red', interpolate=True)

plt.tight_layout()
plt.savefig('../output/figures/03_momentum_strategy_results.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: output/figures/03_momentum_strategy_results.png")

print("\n" + "="*80)
print("MOMENTUM STRATEGY COMPLETE!")
print("="*80)
print(f"\n✓ KEY RESULT: Momentum {'WORKS' if p_value < 0.05 else 'DOES NOT WORK'} in Korean market")
print(f"✓ Long-Short return: {ls_avg:+.4f}% per day ({(1 + ls_avg/100)**252 - 1:+.1%} annualized)")
print(f"✓ Statistical significance: T-stat = {t_stat:.2f}, P-value = {p_value:.4f}\n")
