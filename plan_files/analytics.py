"""
Performance Analytics Module
Calculate and analyze portfolio performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats


class PerformanceAnalytics:
    """
    Comprehensive performance analytics for quantitative strategies
    
    Implements industry-standard metrics used by hedge funds and asset managers
    """
    
    @staticmethod
    def total_return(equity_curve: pd.Series) -> float:
        """
        Calculate total return
        
        Args:
            equity_curve: Series of portfolio values over time
            
        Returns:
            Total return as decimal (e.g., 0.45 = 45%)
        """
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    @staticmethod
    def cagr(equity_curve: pd.Series) -> float:
        """
        Calculate Compound Annual Growth Rate
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            CAGR as decimal
            
        Formula:
            CAGR = (End Value / Start Value)^(1 / Years) - 1
        """
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        if years == 0:
            return 0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    
    @staticmethod
    def volatility(returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)
        
        Args:
            returns: Series of returns
            annualize: If True, annualize the volatility
            
        Returns:
            Volatility as decimal
        """
        vol = returns.std()
        if annualize:
            vol = vol * np.sqrt(252)  # 252 trading days per year
        return vol
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Sharpe ratio
            
        Theory:
            Sharpe = (Return - RiskFree) / Volatility
            Measures risk-adjusted return
            > 1.0 is good, > 2.0 is excellent, > 3.0 is exceptional
            
            William Sharpe won Nobel Prize for this metric (1990)
        """
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0
        sharpe = excess_returns.mean() / excess_returns.std()
        return sharpe * np.sqrt(252)  # Annualize
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino Ratio (like Sharpe but only penalizes downside volatility)
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
            
        Theory:
            Similar to Sharpe but uses downside deviation instead of total volatility
            Better for strategies with positive skew (many small wins, few large losses)
        """
        excess_returns = returns - (risk_free_rate / 252)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
            
        downside_std = downside_returns.std()
        sortino = excess_returns.mean() / downside_std
        return sortino * np.sqrt(252)  # Annualize
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown (worst peak-to-trough decline)
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            Max drawdown as negative decimal (e.g., -0.25 = -25%)
            
        Theory:
            MDD measures the largest cumulative loss from peak
            Critical risk metric - shows worst historical loss
            Investors often more sensitive to drawdowns than volatility
        """
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown.min()
    
    @staticmethod
    def calmar_ratio(equity_curve: pd.Series, returns: pd.Series) -> float:
        """
        Calculate Calmar Ratio (CAGR / |Max Drawdown|)
        
        Args:
            equity_curve: Series of portfolio values
            returns: Series of returns
            
        Returns:
            Calmar ratio
            
        Theory:
            Measures return relative to worst drawdown
            > 0.5 is decent, > 1.0 is good, > 3.0 is exceptional
            Popular in hedge fund industry
        """
        cagr_val = PerformanceAnalytics.cagr(equity_curve)
        max_dd = abs(PerformanceAnalytics.max_drawdown(equity_curve))
        
        if max_dd == 0:
            return 0
        return cagr_val / max_dd
    
    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """
        Calculate percentage of winning days
        
        Args:
            returns: Series of returns
            
        Returns:
            Win rate as decimal (e.g., 0.55 = 55%)
        """
        wins = (returns > 0).sum()
        total = len(returns[returns != 0])
        return wins / total if total > 0 else 0
    
    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Args:
            returns: Series of returns
            
        Returns:
            Profit factor (> 1.0 is profitable)
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 0
        return gains / losses
    
    @staticmethod
    def alpha_beta(strategy_returns: pd.Series,
                   benchmark_returns: pd.Series,
                   risk_free_rate: float = 0.02) -> Tuple[float, float]:
        """
        Calculate alpha and beta relative to benchmark
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Tuple of (alpha, beta)
            
        Theory:
            Beta: Sensitivity to market movements (1.0 = moves with market)
            Alpha: Excess return not explained by market exposure
            
            From CAPM: Return = RiskFree + Beta * (Market - RiskFree) + Alpha
            Alpha > 0 indicates skill-based outperformance
        """
        # Align series
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 0.0, 1.0
        
        # Calculate excess returns
        daily_rf = risk_free_rate / 252
        excess_strategy = aligned['strategy'] - daily_rf
        excess_benchmark = aligned['benchmark'] - daily_rf
        
        # Linear regression: strategy = alpha + beta * benchmark
        covariance = np.cov(excess_strategy, excess_benchmark)[0, 1]
        benchmark_variance = np.var(excess_benchmark)
        
        if benchmark_variance == 0:
            return 0.0, 1.0
            
        beta = covariance / benchmark_variance
        alpha = excess_strategy.mean() - beta * excess_benchmark.mean()
        
        # Annualize alpha
        alpha_annual = alpha * 252
        
        return alpha_annual, beta
    
    @staticmethod
    def information_ratio(strategy_returns: pd.Series,
                         benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio (active return / tracking error)
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
            
        Theory:
            IR measures consistency of outperformance vs benchmark
            > 0.5 is good, > 1.0 is excellent
            Used by institutional investors to evaluate active managers
        """
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        # Active returns (strategy - benchmark)
        active_returns = aligned['strategy'] - aligned['benchmark']
        
        # Tracking error (volatility of active returns)
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        # Information ratio
        ir = active_returns.mean() / tracking_error
        return ir * np.sqrt(252)  # Annualize
    
    @staticmethod
    def drawdown_analysis(equity_curve: pd.Series) -> pd.DataFrame:
        """
        Detailed drawdown analysis
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            DataFrame with drawdown periods
        """
        # Calculate drawdowns
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        
        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        
        if in_drawdown.any():
            # Find start and end of each drawdown period
            starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
            ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)
            
            start_dates = equity_curve.index[starts]
            end_dates = equity_curve.index[ends]
            
            # Handle ongoing drawdown
            if len(start_dates) > len(end_dates):
                end_dates = end_dates.append(pd.Index([equity_curve.index[-1]]))
            
            for start, end in zip(start_dates, end_dates):
                period_dd = drawdowns[start:end]
                max_dd = period_dd.min()
                duration = (end - start).days
                
                drawdown_periods.append({
                    'Start': start,
                    'End': end,
                    'Duration (days)': duration,
                    'Max Drawdown': max_dd,
                    'Recovery': 'Ongoing' if end == equity_curve.index[-1] else 'Recovered'
                })
        
        return pd.DataFrame(drawdown_periods)
    
    @staticmethod
    def rolling_metrics(equity_curve: pd.Series,
                       window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Args:
            equity_curve: Series of portfolio values
            window: Rolling window in days
            
        Returns:
            DataFrame with rolling metrics
        """
        returns = equity_curve.pct_change()
        
        rolling_metrics = pd.DataFrame(index=equity_curve.index)
        
        # Rolling return
        rolling_metrics['Return'] = equity_curve.pct_change(window)
        
        # Rolling volatility
        rolling_metrics['Volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe
        rolling_metrics['Sharpe'] = (
            returns.rolling(window).mean() / returns.rolling(window).std()
        ) * np.sqrt(252)
        
        # Rolling max drawdown
        def rolling_max_dd(series):
            rolling_max = series.expanding().max()
            dd = (series - rolling_max) / rolling_max
            return dd.min()
        
        rolling_metrics['Max DD'] = equity_curve.rolling(window).apply(rolling_max_dd)
        
        return rolling_metrics
    
    @staticmethod
    def generate_report(equity_curve: pd.Series,
                       benchmark: Optional[pd.Series] = None,
                       risk_free_rate: float = 0.02) -> Dict:
        """
        Generate comprehensive performance report
        
        Args:
            equity_curve: Strategy equity curve
            benchmark: Optional benchmark equity curve
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with all performance metrics
        """
        returns = equity_curve.pct_change().dropna()
        
        report = {
            # Return metrics
            'Total Return': PerformanceAnalytics.total_return(equity_curve),
            'CAGR': PerformanceAnalytics.cagr(equity_curve),
            
            # Risk metrics
            'Volatility': PerformanceAnalytics.volatility(returns),
            'Max Drawdown': PerformanceAnalytics.max_drawdown(equity_curve),
            
            # Risk-adjusted returns
            'Sharpe Ratio': PerformanceAnalytics.sharpe_ratio(returns, risk_free_rate),
            'Sortino Ratio': PerformanceAnalytics.sortino_ratio(returns, risk_free_rate),
            'Calmar Ratio': PerformanceAnalytics.calmar_ratio(equity_curve, returns),
            
            # Win/Loss metrics
            'Win Rate': PerformanceAnalytics.win_rate(returns),
            'Profit Factor': PerformanceAnalytics.profit_factor(returns),
            'Avg Win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'Avg Loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
            
            # Trading stats
            'Best Day': returns.max(),
            'Worst Day': returns.min(),
            'Positive Days': (returns > 0).sum(),
            'Negative Days': (returns < 0).sum(),
        }
        
        # Add benchmark comparison if provided
        if benchmark is not None:
            benchmark_returns = benchmark.pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(benchmark_returns.index)
            returns_aligned = returns.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]
            
            alpha, beta = PerformanceAnalytics.alpha_beta(
                returns_aligned, benchmark_aligned, risk_free_rate
            )
            
            report['Alpha'] = alpha
            report['Beta'] = beta
            report['Information Ratio'] = PerformanceAnalytics.information_ratio(
                returns_aligned, benchmark_aligned
            )
            report['Benchmark Return'] = PerformanceAnalytics.total_return(benchmark)
            report['Benchmark CAGR'] = PerformanceAnalytics.cagr(benchmark)
            report['Excess Return'] = report['Total Return'] - report['Benchmark Return']
        
        return report
    
    @staticmethod
    def print_report(report: Dict):
        """
        Pretty print performance report
        
        Args:
            report: Report dictionary from generate_report()
        """
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        print("\n📈 RETURNS")
        print(f"  Total Return:          {report['Total Return']:>10.2%}")
        print(f"  CAGR:                  {report['CAGR']:>10.2%}")
        
        print("\n📊 RISK METRICS")
        print(f"  Volatility:            {report['Volatility']:>10.2%}")
        print(f"  Max Drawdown:          {report['Max Drawdown']:>10.2%}")
        
        print("\n⚡ RISK-ADJUSTED RETURNS")
        print(f"  Sharpe Ratio:          {report['Sharpe Ratio']:>10.2f}")
        print(f"  Sortino Ratio:         {report['Sortino Ratio']:>10.2f}")
        print(f"  Calmar Ratio:          {report['Calmar Ratio']:>10.2f}")
        
        print("\n🎯 WIN/LOSS ANALYSIS")
        print(f"  Win Rate:              {report['Win Rate']:>10.2%}")
        print(f"  Profit Factor:         {report['Profit Factor']:>10.2f}")
        print(f"  Average Win:           {report['Avg Win']:>10.2%}")
        print(f"  Average Loss:          {report['Avg Loss']:>10.2%}")
        
        print("\n📅 TRADING STATISTICS")
        print(f"  Best Day:              {report['Best Day']:>10.2%}")
        print(f"  Worst Day:             {report['Worst Day']:>10.2%}")
        print(f"  Positive Days:         {report['Positive Days']:>10.0f}")
        print(f"  Negative Days:         {report['Negative Days']:>10.0f}")
        
        if 'Alpha' in report:
            print("\n🎖️  BENCHMARK COMPARISON")
            print(f"  Alpha:                 {report['Alpha']:>10.2%}")
            print(f"  Beta:                  {report['Beta']:>10.2f}")
            print(f"  Information Ratio:     {report['Information Ratio']:>10.2f}")
            print(f"  Excess Return:         {report['Excess Return']:>10.2%}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Example usage
    print("Performance Analytics - Example\n")
    
    # Create sample equity curve
    dates = pd.date_range('2020-01-01', '2024-10-25', freq='D')
    np.random.seed(42)
    
    # Simulate returns with positive drift and volatility
    daily_returns = np.random.normal(0.0005, 0.01, len(dates))
    equity_curve = pd.Series(100000 * (1 + daily_returns).cumprod(), index=dates)
    
    # Generate report
    report = PerformanceAnalytics.generate_report(equity_curve)
    
    # Print report
    PerformanceAnalytics.print_report(report)
    
    print("\n✅ Analytics example complete!")
