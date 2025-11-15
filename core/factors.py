"""
Factor Engine for Factor Lab
Calculates quantitative factors from price and fundamental data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorEngine:
    """Calculate quantitative investment factors"""

    @staticmethod
    def calculate_all_factors(prices: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None,
                             tickers: Optional[list] = None) -> pd.DataFrame:
        """
        Calculate all factors for given price and fundamental data

        Args:
            prices: DataFrame with price data (columns should be tickers or MultiIndex)
            fundamentals: DataFrame with fundamental data (indexed by ticker)
            tickers: List of ticker symbols

        Returns:
            DataFrame with factor scores for each ticker
        """
        logger.info("📊 Calculating factors...")

        factors = pd.DataFrame()

        # Extract ticker list from prices if not provided
        if tickers is None:
            if isinstance(prices.columns, pd.MultiIndex):
                tickers = list(prices.columns.get_level_values(0).unique())
            else:
                tickers = list(prices.columns)

        # Get the latest prices for factor calculation
        if isinstance(prices.columns, pd.MultiIndex):
            # Multiple tickers - extract close prices
            close_prices = pd.DataFrame()
            for ticker in tickers:
                if ticker in prices.columns.get_level_values(0):
                    ticker_data = prices[ticker]
                    if 'Close' in ticker_data.columns:
                        close_prices[ticker] = ticker_data['Close']
                    elif 'Adj Close' in ticker_data.columns:
                        close_prices[ticker] = ticker_data['Adj Close']
        else:
            close_prices = prices

        # Calculate price-based factors
        factors['momentum_12m'] = FactorEngine._momentum(close_prices, lookback=252)
        factors['momentum_6m'] = FactorEngine._momentum(close_prices, lookback=126)
        factors['momentum_3m'] = FactorEngine._momentum(close_prices, lookback=63)
        factors['volatility_60d'] = FactorEngine._volatility(close_prices, window=60)

        # Calculate fundamental-based factors if data is available
        if fundamentals is not None and not fundamentals.empty:
            # Value factors
            if 'pb_ratio' in fundamentals.columns:
                factors['value_pb'] = FactorEngine._value_pb(fundamentals)

            if 'pe_ratio' in fundamentals.columns:
                factors['value_pe'] = FactorEngine._value_pe(fundamentals)

            if 'ps_ratio' in fundamentals.columns:
                factors['value_ps'] = FactorEngine._value_ps(fundamentals)

            # Quality factors
            if 'roe' in fundamentals.columns:
                factors['quality_roe'] = FactorEngine._quality_roe(fundamentals)

            if 'roa' in fundamentals.columns:
                factors['quality_roa'] = FactorEngine._quality_roa(fundamentals)

            if 'profit_margin' in fundamentals.columns:
                factors['quality_margin'] = FactorEngine._quality_margin(fundamentals)

            # Size factor
            # For size, we'll use price * volume as a proxy if market_cap not available
            # Lower is better (small cap premium)
            latest_prices = close_prices.iloc[-1]
            if 'market_cap' in fundamentals.columns:
                # Use actual market cap if available
                mcap = fundamentals['market_cap'].reindex(factors.index)
                factors['size_log_mcap'] = -np.log(mcap.replace(0, np.nan))
            else:
                # Estimate from price * average volume
                if len(close_prices) > 20:
                    avg_volume = close_prices.rolling(20).mean().iloc[-1]
                    proxy_mcap = latest_prices * avg_volume
                    factors['size_log_mcap'] = -np.log(proxy_mcap.replace(0, np.nan))

        logger.info(f"✅ Calculated {len(factors.columns)} factors for {len(factors)} stocks")

        return factors

    @staticmethod
    def _momentum(prices: pd.DataFrame, lookback: int = 252, skip_last: int = 21) -> pd.Series:
        """
        Calculate momentum factor (price return over lookback period, skipping last month)

        Academic basis: Jegadeesh & Titman (1993)
        Skip last month to avoid short-term reversal effects

        Args:
            prices: DataFrame of prices (date x ticker)
            lookback: Number of days to look back (252 = 12 months)
            skip_last: Number of days to skip at end (21 = 1 month)

        Returns:
            Series of momentum scores
        """
        if len(prices) < lookback + skip_last:
            logger.warning(f"Not enough data for momentum calculation "
                         f"(need {lookback + skip_last}, have {len(prices)})")
            return pd.Series(np.nan, index=prices.columns)

        # Calculate return from (lookback + skip_last) days ago to skip_last days ago
        start_price = prices.iloc[-(lookback + skip_last)]
        end_price = prices.iloc[-skip_last] if skip_last > 0 else prices.iloc[-1]

        momentum = (end_price / start_price) - 1

        # Replace inf and -inf with NaN
        momentum = momentum.replace([np.inf, -np.inf], np.nan)

        return momentum

    @staticmethod
    def _volatility(prices: pd.DataFrame, window: int = 60) -> pd.Series:
        """
        Calculate volatility factor (lower volatility is better)

        Academic basis: Low-volatility anomaly (Ang et al., 2006)

        Args:
            prices: DataFrame of prices
            window: Rolling window for volatility calculation

        Returns:
            Series of volatility scores (negative, so lower vol = higher score)
        """
        if len(prices) < window:
            logger.warning(f"Not enough data for volatility calculation "
                         f"(need {window}, have {len(prices)})")
            return pd.Series(np.nan, index=prices.columns)

        # Calculate returns
        returns = prices.pct_change()

        # Calculate rolling std (annualized)
        volatility = returns.iloc[-window:].std() * np.sqrt(252)

        # Negative because lower volatility is better
        return -volatility

    @staticmethod
    def _value_pb(fundamentals: pd.DataFrame) -> pd.Series:
        """
        Calculate value factor from Price-to-Book ratio

        Lower P/B = higher value = higher score

        Args:
            fundamentals: DataFrame with 'pb_ratio' column

        Returns:
            Series of value scores
        """
        pb_ratio = fundamentals['pb_ratio']

        # Book-to-Market ratio (inverse of P/B)
        # Higher B/M = more value
        value = 1 / pb_ratio.replace(0, np.nan)

        return value.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _value_pe(fundamentals: pd.DataFrame) -> pd.Series:
        """
        Calculate value factor from Price-to-Earnings ratio

        Lower P/E = higher value = higher score
        """
        pe_ratio = fundamentals['pe_ratio']

        # Earnings-to-Price ratio (inverse of P/E)
        value = 1 / pe_ratio.replace(0, np.nan)

        return value.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _value_ps(fundamentals: pd.DataFrame) -> pd.Series:
        """Calculate value factor from Price-to-Sales ratio"""
        ps_ratio = fundamentals['ps_ratio']
        value = 1 / ps_ratio.replace(0, np.nan)
        return value.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _quality_roe(fundamentals: pd.DataFrame) -> pd.Series:
        """
        Calculate quality factor from Return on Equity

        Academic basis: Novy-Marx (2013)

        Higher ROE = higher quality
        """
        roe = fundamentals['roe']
        return roe.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _quality_roa(fundamentals: pd.DataFrame) -> pd.Series:
        """Calculate quality factor from Return on Assets"""
        roa = fundamentals['roa']
        return roa.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _quality_margin(fundamentals: pd.DataFrame) -> pd.Series:
        """Calculate quality factor from Profit Margin"""
        margin = fundamentals['profit_margin']
        return margin.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def normalize_factors(factors: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize factors using z-score (cross-sectional)

        Each factor will have mean=0, std=1 across stocks

        Args:
            factors: DataFrame of raw factor scores

        Returns:
            DataFrame of normalized factor scores
        """
        normalized = factors.copy()

        for col in factors.columns:
            # Get non-NaN values
            valid_values = factors[col].dropna()

            if len(valid_values) > 1:
                mean = valid_values.mean()
                std = valid_values.std()

                if std > 0:
                    normalized[col] = (factors[col] - mean) / std
                else:
                    # If std is 0, all values are the same, set to 0
                    normalized[col] = 0
            else:
                # Not enough valid values, keep as NaN
                normalized[col] = np.nan

        return normalized

    @staticmethod
    def combine_factors(factors: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """
        Create composite score from weighted combination of factors

        Args:
            factors: DataFrame of normalized factor scores
            weights: Dict like {'momentum_12m': 0.4, 'value_pb': 0.3, ...}

        Returns:
            Series of composite scores
        """
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")

        normalized_weights = {k: v/total_weight for k, v in weights.items()}

        # Initialize composite score
        composite = pd.Series(0.0, index=factors.index)

        # Add weighted factors
        for factor, weight in normalized_weights.items():
            if factor in factors.columns:
                # Only add non-NaN values
                factor_values = factors[factor].fillna(0)
                composite += weight * factor_values
            else:
                logger.warning(f"Factor '{factor}' not found in factors DataFrame")

        return composite

    @staticmethod
    def rank_factors(factors: pd.DataFrame) -> pd.DataFrame:
        """
        Rank factors (percentile-based)

        Args:
            factors: DataFrame of factor scores

        Returns:
            DataFrame of ranked factors (0-1 scale)
        """
        ranked = factors.copy()

        for col in factors.columns:
            # Rank from 0 to 1 (higher is better)
            ranked[col] = factors[col].rank(pct=True)

        return ranked


if __name__ == "__main__":
    # Test factor calculations
    print("\n" + "="*60)
    print("Testing Factor Engine")
    print("="*60 + "\n")

    # Create sample price data
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Simulate prices with different trends
    prices = pd.DataFrame({
        'AAPL': 100 * (1 + np.random.randn(300).cumsum() * 0.01),
        'MSFT': 200 * (1 + np.random.randn(300).cumsum() * 0.01),
        'GOOGL': 150 * (1 + np.random.randn(300).cumsum() * 0.01)
    }, index=dates)

    # Create sample fundamental data
    fundamentals = pd.DataFrame({
        'pb_ratio': [45.3, 12.5, 6.8],
        'pe_ratio': [28.5, 32.1, 25.3],
        'roe': [1.56, 0.42, 0.31],
        'profit_margin': [0.258, 0.362, 0.215]
    }, index=tickers)

    # Test 1: Calculate all factors
    print("1. Calculating factors...")
    factors = FactorEngine.calculate_all_factors(prices, fundamentals, tickers)
    print(f"\nFactors calculated: {factors.columns.tolist()}")
    print("\nRaw factors:")
    print(factors)

    # Test 2: Normalize factors
    print("\n2. Normalizing factors...")
    normalized = FactorEngine.normalize_factors(factors)
    print("\nNormalized factors:")
    print(normalized)
    print(f"\nMean of momentum_12m: {normalized['momentum_12m'].mean():.6f} (should be ~0)")
    print(f"Std of momentum_12m: {normalized['momentum_12m'].std():.6f} (should be ~1)")

    # Test 3: Combine factors
    print("\n3. Combining factors...")
    weights = {
        'momentum_12m': 0.4,
        'value_pb': 0.3,
        'quality_roe': 0.3
    }
    composite = FactorEngine.combine_factors(normalized, weights)
    print("\nComposite scores:")
    print(composite.sort_values(ascending=False))

    # Test 4: Rank factors
    print("\n4. Ranking factors...")
    ranked = FactorEngine.rank_factors(factors)
    print("\nRanked factors (percentiles):")
    print(ranked)

    print("\n✅ Factor engine test complete!")
