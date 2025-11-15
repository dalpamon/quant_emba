"""
Portfolio Construction Module
Implements long-short portfolio strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class Portfolio:
    """
    Portfolio construction for factor-based strategies
    Supports long-short, long-only, and market-neutral approaches
    """
    
    def __init__(self, prices: pd.DataFrame, factor_scores: pd.DataFrame):
        """
        Initialize portfolio constructor
        
        Args:
            prices: DataFrame of adjusted prices (dates x tickers)
            factor_scores: DataFrame of factor scores (dates x tickers)
        """
        self.prices = prices
        self.factor_scores = factor_scores
        
    def create_long_short_portfolio(self, 
                                    date: pd.Timestamp,
                                    top_pct: float = 0.2,
                                    bottom_pct: float = 0.2,
                                    weight_method: str = 'equal') -> Dict:
        """
        Create long-short portfolio based on factor scores
        
        Args:
            date: Rebalancing date
            top_pct: Percentage of stocks to long (e.g., 0.2 = top 20%)
            bottom_pct: Percentage of stocks to short (e.g., 0.2 = bottom 20%)
            weight_method: 'equal', 'score_weighted', 'volatility_weighted'
            
        Returns:
            Dictionary with 'long' and 'short' positions
            
        Theory:
            Long-short portfolios provide several benefits:
            1. Market neutrality (hedged against market movements)
            2. Lower beta exposure
            3. Can profit in both bull and bear markets
            4. Focuses on relative performance (alpha generation)
        """
        
        # Get scores for this date
        if date not in self.factor_scores.index:
            return {'long': {}, 'short': {}}
            
        scores = self.factor_scores.loc[date].dropna()
        
        if len(scores) == 0:
            return {'long': {}, 'short': {}}
        
        # Determine thresholds
        long_threshold = scores.quantile(1 - top_pct)
        short_threshold = scores.quantile(bottom_pct)
        
        # Select stocks
        long_tickers = scores[scores >= long_threshold].index.tolist()
        short_tickers = scores[scores <= short_threshold].index.tolist()
        
        # Calculate weights
        long_weights = self._calculate_weights(
            scores.loc[long_tickers], 
            method=weight_method
        )
        short_weights = self._calculate_weights(
            scores.loc[short_tickers],
            method=weight_method
        )
        
        # Get current prices
        current_prices = self.prices.loc[date] if date in self.prices.index else None
        
        # Build position dictionaries
        long_positions = {}
        for ticker in long_tickers:
            long_positions[ticker] = {
                'weight': long_weights.get(ticker, 0),
                'score': scores.loc[ticker],
                'price': current_prices.loc[ticker] if current_prices is not None else None
            }
        
        short_positions = {}
        for ticker in short_tickers:
            short_positions[ticker] = {
                'weight': short_weights.get(ticker, 0),
                'score': scores.loc[ticker],
                'price': current_prices.loc[ticker] if current_prices is not None else None
            }
        
        return {
            'long': long_positions,
            'short': short_positions,
            'date': date
        }
    
    def _calculate_weights(self, scores: pd.Series, method: str = 'equal') -> Dict[str, float]:
        """
        Calculate position weights
        
        Args:
            scores: Factor scores for selected stocks
            method: Weighting method
            
        Returns:
            Dictionary of {ticker: weight}
        """
        
        if method == 'equal':
            # Equal weight across all positions
            weight = 1.0 / len(scores)
            return {ticker: weight for ticker in scores.index}
            
        elif method == 'score_weighted':
            # Weight proportional to factor score
            abs_scores = scores.abs()
            normalized_scores = abs_scores / abs_scores.sum()
            return normalized_scores.to_dict()
            
        elif method == 'volatility_weighted':
            # Inverse volatility weighting (lower vol = higher weight)
            # This would require volatility data - simplified here
            # In practice, you'd calculate rolling volatility for each stock
            return {ticker: 1.0/len(scores) for ticker in scores.index}
            
        else:
            raise ValueError(f"Unknown weighting method: {method}")
    
    def create_quintile_portfolios(self, date: pd.Timestamp) -> Dict[int, List[str]]:
        """
        Create quintile portfolios (Q1 = lowest, Q5 = highest)
        Useful for analyzing factor spread returns
        
        Args:
            date: Rebalancing date
            
        Returns:
            Dictionary mapping quintile number to list of tickers
        """
        
        if date not in self.factor_scores.index:
            return {}
            
        scores = self.factor_scores.loc[date].dropna()
        
        if len(scores) < 5:
            return {}
        
        # Assign quintiles
        quintiles = pd.qcut(scores, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        portfolios = {}
        for q in [1, 2, 3, 4, 5]:
            portfolios[q] = scores[quintiles == q].index.tolist()
        
        return portfolios
    
    def apply_position_limits(self, positions: Dict, 
                             max_position_size: float = 0.05,
                             max_sector_exposure: Optional[Dict] = None) -> Dict:
        """
        Apply risk management constraints to positions
        
        Args:
            positions: Position dictionary from create_long_short_portfolio
            max_position_size: Maximum weight for any single position
            max_sector_exposure: Dict of {sector: max_weight}
            
        Returns:
            Constrained position dictionary
        """
        
        for position_type in ['long', 'short']:
            if position_type not in positions:
                continue
                
            for ticker, pos_data in positions[position_type].items():
                # Cap individual position size
                if pos_data['weight'] > max_position_size:
                    pos_data['weight'] = max_position_size
            
            # Renormalize weights to sum to 1.0
            total_weight = sum(p['weight'] for p in positions[position_type].values())
            if total_weight > 0:
                for ticker in positions[position_type]:
                    positions[position_type][ticker]['weight'] /= total_weight
        
        return positions
    
    def calculate_turnover(self, 
                          old_positions: Dict, 
                          new_positions: Dict) -> float:
        """
        Calculate portfolio turnover between rebalances
        
        Args:
            old_positions: Previous positions
            new_positions: New positions
            
        Returns:
            Turnover percentage (0-2, where 2 = complete turnover)
            
        Theory:
            Turnover = sum(|new_weight - old_weight|) / 2
            High turnover increases transaction costs
            Target turnover depends on strategy holding period
        """
        
        all_tickers = set()
        for pos_type in ['long', 'short']:
            all_tickers.update(old_positions.get(pos_type, {}).keys())
            all_tickers.update(new_positions.get(pos_type, {}).keys())
        
        turnover = 0
        for ticker in all_tickers:
            old_weight = 0
            new_weight = 0
            
            # Get old weight
            for pos_type in ['long', 'short']:
                if ticker in old_positions.get(pos_type, {}):
                    weight = old_positions[pos_type][ticker]['weight']
                    old_weight += weight if pos_type == 'long' else -weight
            
            # Get new weight
            for pos_type in ['long', 'short']:
                if ticker in new_positions.get(pos_type, {}):
                    weight = new_positions[pos_type][ticker]['weight']
                    new_weight += weight if pos_type == 'long' else -weight
            
            turnover += abs(new_weight - old_weight)
        
        return turnover / 2  # Divide by 2 to get one-way turnover


class RebalancingSchedule:
    """Manage rebalancing schedule for portfolio"""
    
    @staticmethod
    def get_rebalance_dates(start_date: pd.Timestamp,
                           end_date: pd.Timestamp,
                           frequency: str = 'M') -> List[pd.Timestamp]:
        """
        Generate rebalancing dates
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly)
            
        Returns:
            List of rebalancing dates
        """
        
        if frequency == 'D':
            # Daily rebalancing
            dates = pd.date_range(start_date, end_date, freq='B')  # Business days
            
        elif frequency == 'W':
            # Weekly (every Monday)
            dates = pd.date_range(start_date, end_date, freq='W-MON')
            
        elif frequency == 'M':
            # Monthly (last business day)
            dates = pd.date_range(start_date, end_date, freq='BM')
            
        elif frequency == 'Q':
            # Quarterly (last business day of quarter)
            dates = pd.date_range(start_date, end_date, freq='BQ')
            
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        return dates.tolist()


class TransactionCostModel:
    """Model transaction costs for realistic backtesting"""
    
    @staticmethod
    def calculate_costs(turnover: float,
                       portfolio_value: float,
                       cost_bps: float = 10.0,
                       min_cost: float = 1.0) -> float:
        """
        Calculate transaction costs
        
        Args:
            turnover: Portfolio turnover (0-2)
            portfolio_value: Current portfolio value
            cost_bps: Cost in basis points (default: 10 bps = 0.1%)
            min_cost: Minimum cost per trade
            
        Returns:
            Total transaction costs
            
        Theory:
            Transaction costs include:
            1. Commissions (broker fees)
            2. Bid-ask spread
            3. Market impact (moving the price)
            4. Opportunity cost (timing delays)
            
            For retail: 5-20 bps is typical
            For institutional: 2-10 bps
        """
        
        traded_value = turnover * portfolio_value
        cost = (cost_bps / 10000) * traded_value
        
        return max(cost, min_cost)
    
    @staticmethod
    def apply_slippage(fill_price: float,
                      side: str,
                      slippage_bps: float = 5.0) -> float:
        """
        Apply slippage to execution price
        
        Args:
            fill_price: Intended fill price
            side: 'long' or 'short'
            slippage_bps: Slippage in basis points
            
        Returns:
            Adjusted fill price
        """
        
        slippage_factor = slippage_bps / 10000
        
        if side == 'long':
            # Pay more when buying
            return fill_price * (1 + slippage_factor)
        else:
            # Receive less when selling/shorting
            return fill_price * (1 - slippage_factor)


if __name__ == "__main__":
    # Example usage
    print("Portfolio Construction - Example\n")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-10-25', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD']
    
    # Random prices
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    # Random factor scores
    factor_scores = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)),
        index=dates,
        columns=tickers
    )
    
    # Create portfolio
    portfolio = Portfolio(prices, factor_scores)
    
    # Get positions for a specific date
    test_date = dates[100]
    positions = portfolio.create_long_short_portfolio(
        test_date,
        top_pct=0.25,
        bottom_pct=0.25,
        weight_method='equal'
    )
    
    print(f"Portfolio for {test_date.date()}:\n")
    print(f"Long positions ({len(positions['long'])}):")
    for ticker, data in positions['long'].items():
        print(f"  {ticker}: weight={data['weight']:.3f}, score={data['score']:.3f}")
    
    print(f"\nShort positions ({len(positions['short'])}):")
    for ticker, data in positions['short'].items():
        print(f"  {ticker}: weight={data['weight']:.3f}, score={data['score']:.3f}")
    
    # Generate rebalancing schedule
    rebal_dates = RebalancingSchedule.get_rebalance_dates(
        dates[0], dates[-1], frequency='M'
    )
    print(f"\nMonthly rebalancing: {len(rebal_dates)} dates")
    
    print("\n✅ Portfolio construction example complete!")
