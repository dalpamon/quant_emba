"""
Backtesting Engine
Simulates historical portfolio performance with realistic constraints
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from portfolio import Portfolio, TransactionCostModel, RebalancingSchedule


class Backtester:
    """
    Backtest factor-based strategies with realistic assumptions
    
    Features:
    - Long-short portfolio simulation
    - Transaction costs and slippage
    - Rebalancing schedules
    - Performance tracking
    """
    
    def __init__(self, 
                 prices: pd.DataFrame,
                 factor_scores: pd.DataFrame,
                 initial_capital: float = 100000):
        """
        Initialize backtester
        
        Args:
            prices: DataFrame of adjusted prices (dates x tickers)
            factor_scores: DataFrame of factor scores (dates x tickers)
            initial_capital: Starting portfolio value
        """
        self.prices = prices
        self.factor_scores = factor_scores
        self.initial_capital = initial_capital
        self.portfolio_builder = Portfolio(prices, factor_scores)
        
    def run(self,
            start_date: str,
            end_date: str,
            rebalance_freq: str = 'M',
            top_pct: float = 0.2,
            bottom_pct: float = 0.2,
            weight_method: str = 'equal',
            transaction_cost_bps: float = 10.0,
            slippage_bps: float = 5.0) -> Tuple[pd.Series, Dict]:
        """
        Run backtest simulation
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q')
            top_pct: Percentage to long (top performers)
            bottom_pct: Percentage to short (bottom performers)
            weight_method: Position weighting method
            transaction_cost_bps: Transaction costs in basis points
            slippage_bps: Slippage in basis points
            
        Returns:
            Tuple of (equity_curve, positions_history)
        """
        
        print("\n" + "="*60)
        print("RUNNING BACKTEST")
        print("="*60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Rebalance Frequency: {rebalance_freq}")
        print(f"Long/Short: {top_pct*100}% / {bottom_pct*100}%")
        print(f"Transaction Costs: {transaction_cost_bps} bps")
        print("="*60 + "\n")
        
        # Filter dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        prices_subset = self.prices.loc[start_dt:end_dt]
        scores_subset = self.factor_scores.loc[start_dt:end_dt]
        
        # Get rebalancing dates
        rebalance_dates = RebalancingSchedule.get_rebalance_dates(
            start_dt, end_dt, rebalance_freq
        )
        
        # Filter to dates with both prices and scores
        rebalance_dates = [d for d in rebalance_dates 
                          if d in prices_subset.index and d in scores_subset.index]
        
        print(f"Rebalancing on {len(rebalance_dates)} dates\n")
        
        # Initialize tracking
        equity_curve = []
        dates_list = []
        current_positions = {'long': {}, 'short': {}}
        current_value = self.initial_capital
        
        positions_history = {}
        
        # Iterate through all trading days
        for i, date in enumerate(prices_subset.index):
            
            # Check if we need to rebalance
            if date in rebalance_dates:
                print(f"📅 Rebalancing on {date.date()}")
                
                # Generate new positions
                new_positions = self.portfolio_builder.create_long_short_portfolio(
                    date,
                    top_pct=top_pct,
                    bottom_pct=bottom_pct,
                    weight_method=weight_method
                )
                
                # Calculate turnover
                if current_positions['long'] or current_positions['short']:
                    turnover = self.portfolio_builder.calculate_turnover(
                        current_positions, new_positions
                    )
                    print(f"  Turnover: {turnover:.2%}")
                    
                    # Apply transaction costs
                    costs = TransactionCostModel.calculate_costs(
                        turnover, current_value, transaction_cost_bps
                    )
                    current_value -= costs
                    print(f"  Transaction costs: ${costs:,.2f}")
                
                # Update positions
                current_positions = new_positions
                positions_history[date] = new_positions
                
                print(f"  Long: {len(new_positions['long'])} stocks")
                print(f"  Short: {len(new_positions['short'])} stocks")
                print(f"  Portfolio value: ${current_value:,.2f}\n")
            
            # Calculate daily returns
            if i > 0:
                daily_return = self._calculate_portfolio_return(
                    current_positions,
                    prices_subset.iloc[i-1],
                    prices_subset.iloc[i]
                )
                
                current_value *= (1 + daily_return)
            
            # Record
            equity_curve.append(current_value)
            dates_list.append(date)
        
        # Create equity curve series
        equity_series = pd.Series(equity_curve, index=dates_list)
        
        print("="*60)
        print("BACKTEST COMPLETE")
        print(f"Final Value: ${current_value:,.2f}")
        print(f"Total Return: {(current_value/self.initial_capital - 1)*100:.2f}%")
        print("="*60 + "\n")
        
        return equity_series, positions_history
    
    def _calculate_portfolio_return(self,
                                   positions: Dict,
                                   prev_prices: pd.Series,
                                   curr_prices: pd.Series) -> float:
        """
        Calculate portfolio return for one day
        
        Args:
            positions: Current positions dictionary
            prev_prices: Previous day's prices
            curr_prices: Current day's prices
            
        Returns:
            Daily portfolio return
        """
        
        total_return = 0
        
        # Long positions (profit when price goes up)
        for ticker, pos_data in positions['long'].items():
            if ticker in prev_prices and ticker in curr_prices:
                if pd.notna(prev_prices[ticker]) and pd.notna(curr_prices[ticker]):
                    stock_return = (curr_prices[ticker] / prev_prices[ticker]) - 1
                    total_return += pos_data['weight'] * stock_return
        
        # Short positions (profit when price goes down)
        for ticker, pos_data in positions['short'].items():
            if ticker in prev_prices and ticker in curr_prices:
                if pd.notna(prev_prices[ticker]) and pd.notna(curr_prices[ticker]):
                    stock_return = (curr_prices[ticker] / prev_prices[ticker]) - 1
                    total_return -= pos_data['weight'] * stock_return  # Negative because short
        
        return total_return
    
    def run_with_benchmark(self,
                          start_date: str,
                          end_date: str,
                          benchmark_ticker: str = 'SPY',
                          **kwargs) -> Tuple[pd.Series, pd.Series, Dict]:
        """
        Run backtest with benchmark comparison
        
        Args:
            start_date: Start date
            end_date: End date
            benchmark_ticker: Benchmark ticker (e.g., 'SPY' for S&P 500)
            **kwargs: Additional arguments for run()
            
        Returns:
            Tuple of (strategy_equity, benchmark_equity, positions_history)
        """
        
        # Run strategy backtest
        strategy_equity, positions_history = self.run(start_date, end_date, **kwargs)
        
        # Calculate benchmark returns
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        if benchmark_ticker in self.prices.columns:
            benchmark_prices = self.prices[benchmark_ticker].loc[start_dt:end_dt]
            
            # Normalize to initial capital
            benchmark_equity = (benchmark_prices / benchmark_prices.iloc[0]) * self.initial_capital
        else:
            print(f"⚠️  Benchmark {benchmark_ticker} not found in data")
            benchmark_equity = pd.Series(self.initial_capital, index=strategy_equity.index)
        
        return strategy_equity, benchmark_equity, positions_history


class WalkForwardBacktest:
    """
    Walk-forward analysis for more robust strategy evaluation
    
    Theory:
        Walk-forward testing reduces overfitting by:
        1. Training on in-sample period
        2. Testing on out-of-sample period
        3. Rolling forward through time
        
        This simulates real trading where you can't peek into the future
    """
    
    def __init__(self, 
                 prices: pd.DataFrame,
                 factor_scores: pd.DataFrame,
                 initial_capital: float = 100000):
        """
        Initialize walk-forward backtester
        
        Args:
            prices: Price data
            factor_scores: Factor scores
            initial_capital: Starting capital
        """
        self.prices = prices
        self.factor_scores = factor_scores
        self.initial_capital = initial_capital
    
    def run_walk_forward(self,
                        train_months: int = 12,
                        test_months: int = 3,
                        **backtest_params) -> pd.DataFrame:
        """
        Run walk-forward backtest
        
        Args:
            train_months: Training period length
            test_months: Testing period length
            **backtest_params: Parameters for backtest
            
        Returns:
            DataFrame with results for each period
        """
        
        results = []
        
        # Generate train/test windows
        start_date = self.prices.index[0]
        end_date = self.prices.index[-1]
        
        current_start = start_date
        
        while current_start < end_date:
            train_end = current_start + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)
            
            if test_end > end_date:
                break
            
            print(f"\nTrain: {current_start.date()} to {train_end.date()}")
            print(f"Test:  {train_end.date()} to {test_end.date()}")
            
            # Run backtest on test period
            backtester = Backtester(self.prices, self.factor_scores, self.initial_capital)
            equity_curve, _ = backtester.run(
                start_date=train_end.strftime('%Y-%m-%d'),
                end_date=test_end.strftime('%Y-%m-%d'),
                **backtest_params
            )
            
            # Calculate returns
            period_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            
            results.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'test_return': period_return
            })
            
            # Move forward
            current_start = test_end
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Example backtest
    print("Backtesting Engine - Example\n")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-10-25', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'CRM']
    
    # Random walk prices with drift
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 2 + 100,
        index=dates,
        columns=tickers
    )
    prices = prices.clip(lower=10)  # Ensure no negative prices
    
    # Factor scores (random for example)
    factor_scores = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)),
        index=dates,
        columns=tickers
    )
    
    # Run backtest
    backtester = Backtester(prices, factor_scores, initial_capital=100000)
    
    equity_curve, positions = backtester.run(
        start_date='2020-01-01',
        end_date='2024-10-25',
        rebalance_freq='M',
        top_pct=0.3,
        bottom_pct=0.3,
        transaction_cost_bps=10
    )
    
    print(f"Final equity: ${equity_curve.iloc[-1]:,.2f}")
    print(f"Return: {(equity_curve.iloc[-1]/equity_curve.iloc[0] - 1)*100:.2f}%")
    
    print("\n✅ Backtest example complete!")
