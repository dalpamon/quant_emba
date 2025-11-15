"""
Portfolio Construction for Factor Lab
Builds long-short portfolios based on factor scores
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Portfolio:
    """Construct long-short portfolios from factor scores"""

    def __init__(self, long_pct: float = 0.2, short_pct: float = 0.2,
                 portfolio_type: str = 'long-short'):
        """
        Initialize portfolio builder

        Args:
            long_pct: Top X% to go long (e.g., 0.2 = top 20%)
            short_pct: Bottom X% to go short
            portfolio_type: 'long-short' or 'long-only'
        """
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.portfolio_type = portfolio_type

    def construct_portfolio(self, scores: pd.Series) -> Dict[str, List[str]]:
        """
        Create portfolio from composite factor scores

        Args:
            scores: Series of composite factor scores (index=ticker, values=score)

        Returns:
            Dict with 'long' and 'short' lists of tickers
        """
        # Remove NaN values
        valid_scores = scores.dropna()

        if len(valid_scores) == 0:
            logger.warning("No valid scores for portfolio construction")
            return {'long': [], 'short': []}

        # Sort scores (highest is best)
        sorted_scores = valid_scores.sort_values(ascending=False)

        # Calculate number of stocks in long and short
        n_total = len(sorted_scores)
        n_long = max(1, int(n_total * self.long_pct))
        n_short = max(1, int(n_total * self.short_pct))

        # Select long positions (top scores)
        long_tickers = sorted_scores.head(n_long).index.tolist()

        # Select short positions (bottom scores)
        if self.portfolio_type == 'long-short':
            short_tickers = sorted_scores.tail(n_short).index.tolist()
        else:
            short_tickers = []

        logger.info(f"Portfolio: {len(long_tickers)} long, {len(short_tickers)} short")

        return {
            'long': long_tickers,
            'short': short_tickers,
            'scores': sorted_scores
        }

    def get_weights(self, positions: Dict[str, List[str]],
                   weighting: str = 'equal') -> pd.Series:
        """
        Calculate position weights

        Args:
            positions: Dict with 'long' and 'short' lists
            weighting: 'equal' or 'score' (score-weighted)

        Returns:
            Series of weights (ticker -> weight)
        """
        weights = pd.Series(dtype=float)

        long_tickers = positions['long']
        short_tickers = positions['short']

        if weighting == 'equal':
            # Equal weight within long and short legs
            n_long = len(long_tickers)
            n_short = len(short_tickers)

            if n_long > 0:
                long_weight = 0.5 / n_long  # 50% total in long leg
                for ticker in long_tickers:
                    weights[ticker] = long_weight

            if n_short > 0:
                short_weight = -0.5 / n_short  # 50% total in short leg (negative)
                for ticker in short_tickers:
                    weights[ticker] = short_weight

        elif weighting == 'score':
            # Weight by factor score (within each leg)
            scores = positions.get('scores', pd.Series())

            if not scores.empty:
                # Long leg: weight proportional to positive excess score
                long_scores = scores[long_tickers]
                if long_scores.sum() > 0:
                    long_weights = long_scores / long_scores.sum() * 0.5
                    weights = pd.concat([weights, long_weights])

                # Short leg: weight proportional to negative excess score
                short_scores = scores[short_tickers]
                if abs(short_scores.sum()) > 0:
                    short_weights = short_scores / abs(short_scores.sum()) * (-0.5)
                    weights = pd.concat([weights, short_weights])

        return weights

    def calculate_turnover(self, old_weights: pd.Series, new_weights: pd.Series) -> float:
        """
        Calculate portfolio turnover (for transaction cost estimation)

        Args:
            old_weights: Previous period weights
            new_weights: Current period weights

        Returns:
            Turnover as fraction of portfolio
        """
        # Align indices
        all_tickers = set(old_weights.index) | set(new_weights.index)

        old_aligned = pd.Series(0.0, index=list(all_tickers))
        old_aligned.update(old_weights)

        new_aligned = pd.Series(0.0, index=list(all_tickers))
        new_aligned.update(new_weights)

        # Turnover = sum of absolute changes
        turnover = (old_aligned - new_aligned).abs().sum()

        return turnover

    def get_portfolio_stats(self, positions: Dict) -> Dict:
        """Get summary statistics for portfolio"""
        return {
            'n_long': len(positions.get('long', [])),
            'n_short': len(positions.get('short', [])),
            'n_total': len(positions.get('long', [])) + len(positions.get('short', [])),
            'long_pct': self.long_pct,
            'short_pct': self.short_pct,
            'portfolio_type': self.portfolio_type
        }


class PortfolioHistory:
    """Track portfolio positions over time"""

    def __init__(self):
        """Initialize portfolio history tracker"""
        self.positions_history = []
        self.weights_history = []
        self.dates = []

    def add_rebalance(self, date: str, positions: Dict, weights: pd.Series):
        """
        Record a rebalancing event

        Args:
            date: Rebalance date
            positions: Portfolio positions
            weights: Position weights
        """
        self.dates.append(date)
        self.positions_history.append(positions)
        self.weights_history.append(weights)

    def get_position_dates(self) -> List[str]:
        """Get all rebalancing dates"""
        return self.dates

    def get_positions_at_date(self, date: str) -> Dict:
        """Get positions at a specific date"""
        if date in self.dates:
            idx = self.dates.index(date)
            return self.positions_history[idx]
        return {'long': [], 'short': []}

    def get_weights_at_date(self, date: str) -> pd.Series:
        """Get weights at a specific date"""
        if date in self.dates:
            idx = self.dates.index(date)
            return self.weights_history[idx]
        return pd.Series(dtype=float)

    def get_turnover_series(self) -> pd.Series:
        """Calculate turnover at each rebalancing"""
        turnovers = []

        for i in range(1, len(self.weights_history)):
            old_weights = self.weights_history[i-1]
            new_weights = self.weights_history[i]

            # Calculate turnover
            all_tickers = set(old_weights.index) | set(new_weights.index)

            old_aligned = pd.Series(0.0, index=list(all_tickers))
            old_aligned.update(old_weights)

            new_aligned = pd.Series(0.0, index=list(all_tickers))
            new_aligned.update(new_weights)

            turnover = (old_aligned - new_aligned).abs().sum()
            turnovers.append(turnover)

        return pd.Series(turnovers, index=self.dates[1:])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert position history to DataFrame"""
        records = []

        for date, positions, weights in zip(self.dates, self.positions_history,
                                           self.weights_history):
            for ticker in positions.get('long', []):
                records.append({
                    'date': date,
                    'ticker': ticker,
                    'position_type': 'long',
                    'weight': weights.get(ticker, 0),
                    'factor_score': positions.get('scores', pd.Series()).get(ticker, np.nan)
                })

            for ticker in positions.get('short', []):
                records.append({
                    'date': date,
                    'ticker': ticker,
                    'position_type': 'short',
                    'weight': weights.get(ticker, 0),
                    'factor_score': positions.get('scores', pd.Series()).get(ticker, np.nan)
                })

        return pd.DataFrame(records)


if __name__ == "__main__":
    # Test portfolio construction
    print("\n" + "="*60)
    print("Testing Portfolio Construction")
    print("="*60 + "\n")

    # Create sample factor scores
    scores = pd.Series({
        'AAPL': 1.5,
        'MSFT': 1.2,
        'GOOGL': 0.8,
        'AMZN': 0.3,
        'META': -0.2,
        'TSLA': -0.8,
        'NVDA': -1.2,
        'INTC': -1.5
    })

    # Test 1: Long-short portfolio
    print("1. Creating long-short portfolio...")
    portfolio = Portfolio(long_pct=0.25, short_pct=0.25, portfolio_type='long-short')

    positions = portfolio.construct_portfolio(scores)
    print(f"\nLong positions ({len(positions['long'])}): {positions['long']}")
    print(f"Short positions ({len(positions['short'])}): {positions['short']}")

    # Test 2: Get equal weights
    print("\n2. Calculating equal weights...")
    weights = portfolio.get_weights(positions, weighting='equal')
    print("\nWeights:")
    for ticker, weight in weights.items():
        print(f"  {ticker}: {weight:+.4f}")

    print(f"\nTotal long weight: {weights[weights > 0].sum():.4f}")
    print(f"Total short weight: {weights[weights < 0].sum():.4f}")

    # Test 3: Score-weighted
    print("\n3. Calculating score-weighted positions...")
    weights_scored = portfolio.get_weights(positions, weighting='score')
    print("\nScore-weighted:")
    for ticker, weight in weights_scored.items():
        print(f"  {ticker}: {weight:+.4f}")

    # Test 4: Long-only portfolio
    print("\n4. Creating long-only portfolio...")
    portfolio_long = Portfolio(long_pct=0.5, portfolio_type='long-only')
    positions_long = portfolio_long.construct_portfolio(scores)
    print(f"\nLong positions ({len(positions_long['long'])}): {positions_long['long']}")
    print(f"Short positions ({len(positions_long['short'])}): {positions_long['short']}")

    # Test 5: Portfolio history
    print("\n5. Testing portfolio history...")
    history = PortfolioHistory()
    history.add_rebalance('2024-01-31', positions, weights)

    # Simulate a second rebalancing with different scores
    scores2 = scores + np.random.randn(len(scores)) * 0.3
    positions2 = portfolio.construct_portfolio(scores2)
    weights2 = portfolio.get_weights(positions2, weighting='equal')
    history.add_rebalance('2024-02-29', positions2, weights2)

    print(f"\nRebalance dates: {history.get_position_dates()}")

    turnovers = history.get_turnover_series()
    print(f"Turnover at 2024-02-29: {turnovers.iloc[0]:.2%}")

    df = history.to_dataframe()
    print(f"\nPosition history shape: {df.shape}")
    print(df.head())

    print("\n✅ Portfolio construction test complete!")
