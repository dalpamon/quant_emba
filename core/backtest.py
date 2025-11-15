"""
Backtesting Engine for Factor Lab
Simulates portfolio performance over time with transaction costs
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """Vectorized backtesting engine for factor strategies"""

    def __init__(self, initial_capital: float = 100000,
                 transaction_cost_bps: float = 10,
                 rebalance_freq: str = 'M'):
        """
        Initialize backtester

        Args:
            initial_capital: Starting portfolio value ($)
            transaction_cost_bps: Transaction cost in basis points (10 bps = 0.1%)
            rebalance_freq: Rebalancing frequency ('M'=monthly, 'Q'=quarterly, 'Y'=yearly)
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.transaction_cost = transaction_cost_bps / 10000
        self.rebalance_freq = rebalance_freq

    def run(self, prices: pd.DataFrame, positions_by_date: Dict[str, Dict],
           weights_by_date: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """
        Run backtest simulation

        Args:
            prices: DataFrame of prices (date x ticker) or MultiIndex columns (ticker, field)
            positions_by_date: Dict {date_str: {'long': [...], 'short': [...]}}
            weights_by_date: Dict {date_str: pd.Series of weights} (optional, uses equal weight if None)

        Returns:
            DataFrame with daily portfolio values and returns
        """
        logger.info(f"\n🚀 Running backtest...")
        logger.info(f"  Period: {prices.index[0].date()} to {prices.index[-1].date()}")
        logger.info(f"  Initial capital: ${self.initial_capital:,.0f}")
        logger.info(f"  Transaction cost: {self.transaction_cost_bps} bps")
        logger.info(f"  Rebalancing: {self.rebalance_freq}")

        # Extract close prices
        close_prices = self._extract_close_prices(prices)

        # Calculate daily returns
        returns = close_prices.pct_change()

        # Initialize tracking variables
        portfolio_values = [self.initial_capital]
        dates_list = [close_prices.index[0]]
        daily_returns = [0.0]

        current_weights = pd.Series(dtype=float)
        rebalance_dates = sorted(positions_by_date.keys())

        logger.info(f"  Rebalancing dates: {len(rebalance_dates)}")

        # Iterate through each trading day
        for i in range(1, len(close_prices)):
            current_date = close_prices.index[i]
            current_date_str = current_date.strftime('%Y-%m-%d')

            # Check if we need to rebalance
            if current_date_str in rebalance_dates:
                # Get new positions
                positions = positions_by_date[current_date_str]

                # Get weights (use provided weights or equal weight)
                if weights_by_date and current_date_str in weights_by_date:
                    new_weights = weights_by_date[current_date_str]
                else:
                    # Equal weight
                    new_weights = self._calculate_equal_weights(positions)

                # Calculate turnover
                turnover = self._calculate_turnover(current_weights, new_weights)

                # Apply transaction costs
                cost = portfolio_values[-1] * self.transaction_cost * turnover

                # Update portfolio value (subtract costs)
                portfolio_values[-1] -= cost

                # Update weights
                current_weights = new_weights

                logger.debug(f"  Rebalanced on {current_date_str}: turnover={turnover:.2%}, cost=${cost:.2f}")

            # Calculate portfolio return for the day
            if not current_weights.empty:
                # Get returns for stocks we hold
                available_tickers = [t for t in current_weights.index if t in returns.columns]

                if available_tickers:
                    stock_returns = returns.loc[current_date, available_tickers]
                    portfolio_weights = current_weights[available_tickers]

                    # Portfolio return = weighted sum of stock returns
                    portfolio_return = (stock_returns * portfolio_weights).sum()

                    # Handle NaN returns (missing data)
                    if pd.isna(portfolio_return):
                        portfolio_return = 0.0
                else:
                    portfolio_return = 0.0
            else:
                # No positions, no return
                portfolio_return = 0.0

            # Update portfolio value
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
            dates_list.append(current_date)
            daily_returns.append(portfolio_return)

        # Create results DataFrame
        results = pd.DataFrame({
            'date': dates_list,
            'portfolio_value': portfolio_values,
            'daily_return': daily_returns
        })
        results['date'] = pd.to_datetime(results['date'])
        results.set_index('date', inplace=True)

        # Calculate cumulative return
        results['cumulative_return'] = (results['portfolio_value'] / self.initial_capital) - 1

        # Calculate drawdown
        results['drawdown'] = self._calculate_drawdown(results['portfolio_value'])

        final_value = portfolio_values[-1]
        total_return = (final_value / self.initial_capital) - 1

        logger.info(f"\n✅ Backtest complete")
        logger.info(f"  Final value: ${final_value:,.2f}")
        logger.info(f"  Total return: {total_return:.2%}")
        logger.info(f"  Days: {len(results)}")

        return results

    def _extract_close_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Extract close prices from price DataFrame"""
        if isinstance(prices.columns, pd.MultiIndex):
            # MultiIndex columns: (ticker, field)
            close_prices = pd.DataFrame()

            for ticker in prices.columns.get_level_values(0).unique():
                ticker_data = prices[ticker]
                if 'Close' in ticker_data.columns:
                    close_prices[ticker] = ticker_data['Close']
                elif 'Adj Close' in ticker_data.columns:
                    close_prices[ticker] = ticker_data['Adj Close']

            return close_prices
        else:
            # Already in ticker x date format
            return prices

    def _calculate_equal_weights(self, positions: Dict) -> pd.Series:
        """Calculate equal weights for portfolio positions"""
        weights = pd.Series(dtype=float)

        long_tickers = positions.get('long', [])
        short_tickers = positions.get('short', [])

        n_long = len(long_tickers)
        n_short = len(short_tickers)

        if n_long > 0:
            long_weight = 0.5 / n_long
            for ticker in long_tickers:
                weights[ticker] = long_weight

        if n_short > 0:
            short_weight = -0.5 / n_short
            for ticker in short_tickers:
                weights[ticker] = short_weight

        return weights

    def _calculate_turnover(self, old_weights: pd.Series, new_weights: pd.Series) -> float:
        """Calculate portfolio turnover"""
        # Get all tickers
        all_tickers = set(old_weights.index) | set(new_weights.index)

        # Align weights
        old_aligned = pd.Series(0.0, index=list(all_tickers))
        old_aligned.update(old_weights)

        new_aligned = pd.Series(0.0, index=list(all_tickers))
        new_aligned.update(new_weights)

        # Turnover = sum of absolute changes
        turnover = (old_aligned - new_aligned).abs().sum()

        return turnover

    def _calculate_drawdown(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()

        # Drawdown = (value - running_max) / running_max
        drawdown = (portfolio_values - running_max) / running_max

        return drawdown

    def run_multiple_strategies(self, prices: pd.DataFrame,
                               strategies: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Run multiple strategies for comparison

        Args:
            prices: Price data
            strategies: Dict {strategy_name: {positions_by_date, weights_by_date}}

        Returns:
            Dict {strategy_name: results_df}
        """
        logger.info(f"\n🔄 Running {len(strategies)} strategies...")

        results = {}

        for name, strategy_data in strategies.items():
            logger.info(f"\n  Strategy: {name}")
            results[name] = self.run(
                prices,
                strategy_data['positions_by_date'],
                strategy_data.get('weights_by_date')
            )

        return results


if __name__ == "__main__":
    # Test backtester
    print("\n" + "="*60)
    print("Testing Backtest Engine")
    print("="*60 + "\n")

    # Create sample price data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    # Simulate prices with random walks
    np.random.seed(42)
    prices_data = {}
    for ticker in tickers:
        # Random walk with drift
        returns = np.random.randn(252) * 0.015 + 0.0005
        prices = 100 * np.exp(returns.cumsum())
        prices_data[ticker] = prices

    prices = pd.DataFrame(prices_data, index=dates)

    print("Sample prices:")
    print(prices.head())
    print(f"\nPrice data shape: {prices.shape}")

    # Create sample rebalancing schedule (monthly)
    month_ends = prices.resample('M').last().index

    # Create sample positions (rotate between different stocks)
    positions_by_date = {}

    for i, date in enumerate(month_ends):
        date_str = date.strftime('%Y-%m-%d')

        # Rotate positions to simulate rebalancing
        if i % 2 == 0:
            positions_by_date[date_str] = {
                'long': ['AAPL', 'MSFT'],
                'short': ['GOOGL']
            }
        else:
            positions_by_date[date_str] = {
                'long': ['GOOGL', 'AMZN'],
                'short': ['META']
            }

    print(f"\nRebalancing dates: {len(positions_by_date)}")

    # Run backtest
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost_bps=10,
        rebalance_freq='M'
    )

    results = backtester.run(prices, positions_by_date)

    print("\nBacktest results:")
    print(results.head())
    print("\nFinal results:")
    print(results.tail())

    print(f"\nSummary statistics:")
    print(f"  Initial value: $100,000")
    print(f"  Final value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"  Total return: {results['cumulative_return'].iloc[-1]:.2%}")
    print(f"  Max drawdown: {results['drawdown'].min():.2%}")
    print(f"  Daily return (mean): {results['daily_return'].mean():.4%}")
    print(f"  Daily return (std): {results['daily_return'].std():.4%}")

    print("\n✅ Backtest engine test complete!")
