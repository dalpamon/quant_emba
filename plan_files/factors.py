"""
Factor Engine for Quantitative Analysis
Implements common factor calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class FactorEngine:
    """
    Calculate and manipulate quantitative factors
    
    Supported factors:
    - Momentum (various lookback periods)
    - Value (P/B, P/E, P/S ratios)
    - Quality (ROE, ROA, Profitability)
    - Size (Market cap)
    - Volatility (Standard deviation of returns)
    - Liquidity (Trading volume)
    """
    
    @staticmethod
    def momentum(prices: pd.DataFrame, lookback: int = 252, skip_periods: int = 21) -> pd.Series:
        """
        Calculate momentum factor
        
        Args:
            prices: DataFrame or Series of prices
            lookback: Lookback period in days (default: 252 = 1 year)
            skip_periods: Days to skip before measurement (default: 21 = 1 month)
                         Skipping recent period avoids short-term reversal effects
        
        Returns:
            Momentum as percentage change
            
        Theory:
            Momentum effect discovered by Jegadeesh & Titman (1993)
            Stocks with strong past performance tend to continue performing well
            Skip recent month to avoid short-term reversal
        """
        return prices.pct_change(lookback).shift(skip_periods)
    
    @staticmethod
    def volatility(returns: pd.Series, window: int = 60, annualize: bool = True) -> pd.Series:
        """
        Calculate rolling volatility (idiosyncratic risk)
        
        Args:
            returns: Series of daily returns
            window: Rolling window in days (default: 60)
            annualize: If True, annualize the volatility
            
        Returns:
            Rolling standard deviation of returns
            
        Theory:
            Low volatility anomaly (Ang et al., 2006)
            Low volatility stocks tend to outperform high volatility stocks
            This contradicts traditional CAPM theory
        """
        vol = returns.rolling(window).std()
        if annualize:
            vol = vol * np.sqrt(252)  # Annualize using sqrt(trading days)
        return vol
    
    @staticmethod
    def value_btm(pb_ratio: pd.Series) -> pd.Series:
        """
        Calculate Book-to-Market (Value factor)
        
        Args:
            pb_ratio: Price-to-Book ratio
            
        Returns:
            Book-to-Market ratio (inverse of P/B)
            
        Theory:
            Value premium documented by Fama & French (1992)
            High B/M (low P/B) stocks tend to outperform
            Captures value vs growth dimension
        """
        return 1 / pb_ratio.replace(0, np.nan)
    
    @staticmethod
    def value_ep(pe_ratio: pd.Series) -> pd.Series:
        """
        Calculate Earnings-to-Price (Value factor)
        
        Args:
            pe_ratio: Price-to-Earnings ratio
            
        Returns:
            Earnings-to-Price ratio (inverse of P/E)
        """
        return 1 / pe_ratio.replace(0, np.nan)
    
    @staticmethod
    def size(market_cap: pd.Series, log_transform: bool = True) -> pd.Series:
        """
        Calculate size factor
        
        Args:
            market_cap: Market capitalization
            log_transform: If True, return log of market cap
            
        Returns:
            Size measure (log or raw market cap)
            
        Theory:
            Small cap premium (Banz, 1981; Fama & French, 1992)
            Smaller companies tend to outperform larger ones
            Log transform makes the distribution more normal
        """
        if log_transform:
            return np.log(market_cap.replace(0, np.nan))
        return market_cap
    
    @staticmethod
    def quality_roe(roe: pd.Series) -> pd.Series:
        """
        Quality factor based on Return on Equity
        
        Args:
            roe: Return on Equity
            
        Returns:
            ROE as quality measure
            
        Theory:
            Quality premium (Novy-Marx, 2013)
            Profitable companies with strong fundamentals outperform
            ROE measures how efficiently a company uses equity
        """
        return roe
    
    @staticmethod
    def quality_profitability(profit_margin: pd.Series) -> pd.Series:
        """
        Quality factor based on profit margins
        
        Args:
            profit_margin: Net profit margin
            
        Returns:
            Profit margin as quality measure
        """
        return profit_margin
    
    @staticmethod
    def composite_score(factor_dict: Dict[str, pd.Series], 
                       weights: Dict[str, float],
                       standardize: bool = True) -> pd.Series:
        """
        Create composite factor score by combining multiple factors
        
        Args:
            factor_dict: Dictionary of {factor_name: factor_series}
            weights: Dictionary of {factor_name: weight}
            standardize: If True, z-score standardize each factor first
            
        Returns:
            Composite score combining all factors
            
        Example:
            >>> factors = {
            ...     'momentum': momentum_series,
            ...     'value': value_series,
            ...     'quality': quality_series
            ... }
            >>> weights = {'momentum': 0.4, 'value': 0.3, 'quality': 0.3}
            >>> score = FactorEngine.composite_score(factors, weights)
        """
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        composite = pd.Series(0, index=factor_dict[list(factor_dict.keys())[0]].index)
        
        for factor_name, factor_series in factor_dict.items():
            if factor_name not in weights:
                continue
                
            # Standardize factor (z-score)
            if standardize:
                factor_standardized = (factor_series - factor_series.mean()) / factor_series.std()
            else:
                factor_standardized = factor_series
            
            # Add weighted factor to composite
            composite += weights[factor_name] * factor_standardized
        
        return composite
    
    @staticmethod
    def rank_signals(scores: pd.Series, method: str = 'percentile') -> pd.Series:
        """
        Rank signals for portfolio construction
        
        Args:
            scores: Factor scores to rank
            method: Ranking method ('percentile', 'z-score', 'quantile')
            
        Returns:
            Ranked scores
        """
        
        if method == 'percentile':
            # Convert to percentile ranks (0-100)
            return scores.rank(pct=True) * 100
            
        elif method == 'z-score':
            # Standardize to z-scores
            return (scores - scores.mean()) / scores.std()
            
        elif method == 'quantile':
            # Assign quintile ranks (1-5)
            return pd.qcut(scores, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            
        else:
            raise ValueError(f"Unknown ranking method: {method}")
    
    @staticmethod
    def winsorize(series: pd.Series, lower_pct: float = 0.05, 
                 upper_pct: float = 0.95) -> pd.Series:
        """
        Winsorize extreme values to reduce outlier impact
        
        Args:
            series: Data series
            lower_pct: Lower percentile to clip
            upper_pct: Upper percentile to clip
            
        Returns:
            Winsorized series
        """
        lower_bound = series.quantile(lower_pct)
        upper_bound = series.quantile(upper_pct)
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    @staticmethod
    def neutralize_factor(factor: pd.Series, 
                         neutralize_by: pd.Series) -> pd.Series:
        """
        Orthogonalize a factor with respect to another factor
        Useful for creating market-neutral or sector-neutral factors
        
        Args:
            factor: Factor to neutralize
            neutralize_by: Factor to neutralize against (e.g., market beta, sector)
            
        Returns:
            Residuals (neutralized factor)
        """
        from sklearn.linear_model import LinearRegression
        
        # Remove NaN values
        valid_idx = factor.notna() & neutralize_by.notna()
        X = neutralize_by[valid_idx].values.reshape(-1, 1)
        y = factor[valid_idx].values
        
        # Fit linear model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get residuals
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Create series with NaN for invalid indices
        result = pd.Series(np.nan, index=factor.index)
        result.loc[valid_idx] = residuals
        
        return result


class FactorAnalyzer:
    """Analyze factor performance and characteristics"""
    
    @staticmethod
    def factor_summary(factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for factors
        
        Args:
            factor_df: DataFrame with factor columns
            
        Returns:
            Summary statistics DataFrame
        """
        
        summary = pd.DataFrame({
            'Mean': factor_df.mean(),
            'Median': factor_df.median(),
            'Std': factor_df.std(),
            'Min': factor_df.min(),
            'Max': factor_df.max(),
            'Skew': factor_df.skew(),
            'Kurt': factor_df.kurtosis(),
            'Missing %': (factor_df.isna().sum() / len(factor_df)) * 100
        })
        
        return summary.round(4)
    
    @staticmethod
    def factor_correlation(factor_df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix between factors
        
        Args:
            factor_df: DataFrame with factor columns
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            Correlation matrix
        """
        return factor_df.corr(method=method)
    
    @staticmethod
    def factor_turnover(factor_ranks: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate factor turnover (how often ranks change)
        
        Args:
            factor_ranks: DataFrame with ranked factors over time
            window: Window for calculating turnover
            
        Returns:
            Series of turnover percentages
        """
        # Calculate percentage of rank changes
        rank_changes = (factor_ranks != factor_ranks.shift(1)).sum(axis=1)
        turnover = rank_changes / len(factor_ranks.columns)
        
        return turnover.rolling(window).mean()


if __name__ == "__main__":
    # Example usage
    print("Factor Engine - Example Calculations\n")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-10-25', freq='D')
    n_stocks = 10
    
    # Random walk prices
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(len(dates), n_stocks).cumsum(axis=0) + 100,
        index=dates,
        columns=[f'Stock_{i}' for i in range(n_stocks)]
    )
    
    # Calculate momentum for one stock
    momentum_12m = FactorEngine.momentum(prices['Stock_0'], lookback=252, skip_periods=21)
    print(f"Momentum (12-month):\n{momentum_12m.tail()}\n")
    
    # Calculate volatility
    returns = prices.pct_change()
    vol = FactorEngine.volatility(returns['Stock_0'], window=60)
    print(f"Volatility (60-day):\n{vol.tail()}\n")
    
    # Composite score example
    factor_dict = {
        'momentum': momentum_12m,
        'volatility': -vol  # Negative because we want low volatility
    }
    weights = {'momentum': 0.6, 'volatility': 0.4}
    
    composite = FactorEngine.composite_score(factor_dict, weights)
    print(f"Composite Score:\n{composite.tail()}\n")
    
    print("✅ Factor calculations complete!")
