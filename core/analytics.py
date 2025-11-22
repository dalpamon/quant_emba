"""
Performance Analytics for Factor Lab
Calculate risk-adjusted returns and other performance metrics
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """
    Calculate comprehensive performance metrics for backtested strategies.

    Enhanced with:
    - Geometric vs. Arithmetic returns
    - Korean market-specific metrics
    - Advanced risk analytics
    """

    @staticmethod
    def calculate_metrics(equity_curve: pd.Series, benchmark: Optional[pd.Series] = None,
                         risk_free_rate: float = 0.02, korean_market: bool = False) -> Dict:
        """
        Calculate all performance metrics

        Args:
            equity_curve: Series of portfolio values over time
            benchmark: Optional benchmark series for comparison
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Dict of performance metrics
        """
        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Time period
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25

        # Return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Sharpe ratio
        excess_returns = returns - (risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Max drawdown
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Calmar ratio (CAGR / |Max Drawdown|)
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win/Loss statistics
        win_rate = (returns > 0).sum() / len(returns)
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_periods': len(returns),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'years': years
        }

        # Add benchmark comparison if provided
        if benchmark is not None:
            benchmark_metrics = PerformanceAnalytics.calculate_benchmark_metrics(
                equity_curve, benchmark, risk_free_rate
            )
            metrics.update(benchmark_metrics)

        return metrics

    @staticmethod
    def calculate_benchmark_metrics(strategy: pd.Series, benchmark: pd.Series,
                                   risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate metrics relative to benchmark

        Args:
            strategy: Strategy equity curve
            benchmark: Benchmark equity curve
            risk_free_rate: Annual risk-free rate

        Returns:
            Dict of relative metrics
        """
        # Align indices
        common_dates = strategy.index.intersection(benchmark.index)
        strategy_aligned = strategy.loc[common_dates]
        benchmark_aligned = benchmark.loc[common_dates]

        # Calculate returns
        strategy_returns = strategy_aligned.pct_change().dropna()
        benchmark_returns = benchmark_aligned.pct_change().dropna()

        # Active return
        active_returns = strategy_returns - benchmark_returns

        # Tracking error
        tracking_error = active_returns.std() * np.sqrt(252)

        # Information ratio
        information_ratio = (active_returns.mean() * 252 / tracking_error) if tracking_error > 0 else 0

        # Beta
        covariance = strategy_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Alpha (using CAPM)
        rf_daily = risk_free_rate / 252
        expected_return = rf_daily + beta * (benchmark_returns.mean() - rf_daily)
        alpha = strategy_returns.mean() - expected_return
        alpha_annualized = alpha * 252

        # Benchmark metrics
        benchmark_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) - 1
        years = (benchmark_aligned.index[-1] - benchmark_aligned.index[0]).days / 365.25
        benchmark_cagr = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) ** (1/years) - 1

        # Excess return
        strategy_cagr = (strategy_aligned.iloc[-1] / strategy_aligned.iloc[0]) ** (1/years) - 1
        excess_return = strategy_cagr - benchmark_cagr

        return {
            'beta': beta,
            'alpha_annualized': alpha_annualized,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'benchmark_return': benchmark_return,
            'benchmark_cagr': benchmark_cagr,
            'excess_return': excess_return
        }

    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics

        Args:
            returns: Series of returns
            window: Rolling window size (default 252 = 1 year)

        Returns:
            DataFrame with rolling metrics
        """
        rolling = pd.DataFrame(index=returns.index)

        # Rolling Sharpe ratio
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling['sharpe'] = np.sqrt(252) * rolling_mean / rolling_std

        # Rolling volatility
        rolling['volatility'] = rolling_std * np.sqrt(252)

        # Rolling max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window, min_periods=1).max()
        rolling_dd = (cumulative - rolling_max) / rolling_max
        rolling['max_drawdown'] = rolling_dd.rolling(window).min()

        return rolling

    @staticmethod
    def print_report(metrics: Dict, title: str = "Performance Report"):
        """
        Print formatted performance report

        Args:
            metrics: Dict of performance metrics
            title: Report title
        """
        print("\n" + "="*60)
        print(f"{title}")
        print("="*60)

        print(f"\nPeriod: {metrics['start_date']} to {metrics['end_date']} ({metrics['years']:.1f} years)")

        print("\n📊 RETURN METRICS")
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  CAGR:                {metrics['cagr']:>10.2%}")

        if 'benchmark_cagr' in metrics:
            print(f"  Benchmark CAGR:      {metrics['benchmark_cagr']:>10.2%}")
            print(f"  Excess Return:       {metrics['excess_return']:>10.2%}")

        print("\n📉 RISK METRICS")
        print(f"  Volatility:          {metrics['volatility']:>10.2%}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")

        if 'beta' in metrics:
            print(f"  Beta:                {metrics['beta']:>10.2f}")
            print(f"  Tracking Error:      {metrics['tracking_error']:>10.2%}")

        print("\n⚖️  RISK-ADJUSTED RETURNS")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")

        if 'information_ratio' in metrics:
            print(f"  Information Ratio:   {metrics['information_ratio']:>10.2f}")

        if 'alpha_annualized' in metrics:
            print(f"  Alpha (annualized):  {metrics['alpha_annualized']:>10.2%}")

        print("\n💰 WIN/LOSS ANALYSIS")
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Avg Win:             {metrics['avg_win']:>10.4%}")
        print(f"  Avg Loss:            {metrics['avg_loss']:>10.4%}")
        print(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")

        print("\n" + "="*60)

    @staticmethod
    def calculate_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns table"""
        returns = equity_curve.pct_change()
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create year x month table
        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month

        pivot = monthly_df.pivot(index='year', columns='month', values='return')

        # Add yearly totals
        yearly = equity_curve.resample('Y').last().pct_change()
        pivot['Year'] = yearly.values

        return pivot

    @staticmethod
    def calculate_drawdown_periods(equity_curve: pd.Series, top_n: int = 5) -> pd.DataFrame:
        """
        Find the worst drawdown periods

        Args:
            equity_curve: Series of portfolio values
            top_n: Number of worst periods to return

        Returns:
            DataFrame with drawdown periods
        """
        # Calculate drawdowns
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_periods = []

        start_dd = None
        for date, is_dd in in_drawdown.items():
            if is_dd and start_dd is None:
                # Start of new drawdown
                start_dd = date
            elif not is_dd and start_dd is not None:
                # End of drawdown
                dd_series = drawdowns[start_dd:date]
                drawdown_periods.append({
                    'start': start_dd,
                    'end': date,
                    'depth': dd_series.min(),
                    'length_days': (date - start_dd).days,
                    'recovery': date
                })
                start_dd = None

        # Sort by depth and take top N
        drawdown_periods.sort(key=lambda x: x['depth'])
        top_periods = drawdown_periods[:top_n]

        return pd.DataFrame(top_periods)

    @staticmethod
    def geometric_return(returns: pd.Series, periods: int = 252) -> float:
        """
        Calculate geometric (compounded) return.

        More accurate than arithmetic mean for volatile returns.

        Args:
            returns: Series of returns
            periods: Periods per year for annualization (default 252)

        Returns:
            Annualized geometric return
        """
        if len(returns) == 0:
            return 0.0

        # Compound returns
        total_return = (1 + returns).prod() - 1

        # Annualize
        n_periods = len(returns)
        years = n_periods / periods

        if years > 0:
            geometric_return = (1 + total_return) ** (1 / years) - 1
        else:
            geometric_return = 0.0

        return geometric_return

    @staticmethod
    def arithmetic_return(returns: pd.Series, periods: int = 252) -> float:
        """
        Calculate arithmetic (simple average) return.

        Args:
            returns: Series of returns
            periods: Periods per year for annualization (default 252)

        Returns:
            Annualized arithmetic return
        """
        if len(returns) == 0:
            return 0.0

        return returns.mean() * periods

    @staticmethod
    def return_comparison(returns: pd.Series, periods: int = 252) -> Dict:
        """
        Compare geometric vs. arithmetic returns.

        For volatile assets, geometric return is more accurate.
        The difference indicates compounding drag from volatility.

        Args:
            returns: Series of returns
            periods: Periods per year for annualization

        Returns:
            Dict with geometric, arithmetic returns and variance drag
        """
        geometric = PerformanceAnalytics.geometric_return(returns, periods)
        arithmetic = PerformanceAnalytics.arithmetic_return(returns, periods)

        # Variance drag (approximate)
        # geometric ≈ arithmetic - (variance / 2)
        variance = returns.var() * periods
        variance_drag = arithmetic - geometric

        return {
            'geometric_return': geometric,
            'arithmetic_return': arithmetic,
            'variance_drag': variance_drag,
            'variance': variance,
            'note': 'Geometric return accounts for compounding effects'
        }

    @staticmethod
    def korean_market_metrics(strategy_returns: pd.Series, market_returns: pd.Series) -> Dict:
        """
        Calculate Korean market-specific performance metrics.

        Accounts for:
        - Mean reversion tendency in Korean market
        - Momentum reversals
        - Chaebols influence
        - Retail investor sentiment

        Args:
            strategy_returns: Strategy returns
            market_returns: Market (KOSPI) returns

        Returns:
            Dict of Korean market metrics
        """
        # Mean reversion test: autocorrelation
        # Negative autocorrelation suggests mean reversion
        autocorr_1d = strategy_returns.autocorr(lag=1)
        autocorr_5d = strategy_returns.autocorr(lag=5)
        autocorr_20d = strategy_returns.autocorr(lag=20)

        # Momentum reversal: correlation between past and future returns
        # If negative, winners become losers (mean reversion)
        past_returns = strategy_returns.shift(20)
        future_returns = strategy_returns.shift(-20)
        momentum_reversal = past_returns.corr(future_returns)

        # Market correlation during up/down markets
        # Korean retail investors tend to chase performance
        up_market = market_returns > 0
        down_market = market_returns < 0

        corr_up_market = strategy_returns[up_market].corr(market_returns[up_market])
        corr_down_market = strategy_returns[down_market].corr(market_returns[down_market])

        # Tail risk: Korean market can have extreme moves
        # Skewness and kurtosis
        skewness = strategy_returns.skew()
        kurtosis = strategy_returns.kurtosis()

        # Downside risk in tail events (5th percentile)
        var_5 = strategy_returns.quantile(0.05)  # Value at Risk
        cvar_5 = strategy_returns[strategy_returns <= var_5].mean()  # Conditional VaR

        return {
            'mean_reversion_1d': autocorr_1d,
            'mean_reversion_5d': autocorr_5d,
            'mean_reversion_20d': autocorr_20d,
            'momentum_reversal': momentum_reversal,
            'correlation_up_market': corr_up_market,
            'correlation_down_market': corr_down_market,
            'asymmetric_correlation': corr_down_market - corr_up_market,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_5pct': var_5,
            'cvar_5pct': cvar_5,
            'interpretation': {
                'mean_reversion': 'Negative autocorr suggests mean reversion (common in Korean market)',
                'momentum_reversal': 'Negative value indicates winners become losers',
                'asymmetric_correlation': 'Positive value indicates higher correlation in down markets',
                'tail_risk': f'Kurtosis {kurtosis:.2f} (>3 means fat tails, common in Korean market)'
            }
        }

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods: int = 252) -> float:
        """
        Calculate Sharpe ratio using geometric returns.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods: Periods per year

        Returns:
            Sharpe ratio
        """
        if returns.std() == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods)
        sharpe = np.sqrt(periods) * excess_returns.mean() / returns.std()

        return sharpe

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods: int = 252) -> float:
        """
        Calculate Sortino ratio (only downside deviation).

        Better than Sharpe for asymmetric returns.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods: Periods per year

        Returns:
            Sortino ratio
        """
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        excess_returns = returns.mean() - (risk_free_rate / periods)
        downside_std = downside_returns.std() * np.sqrt(periods)

        sortino = excess_returns * periods / downside_std

        return sortino

    @staticmethod
    def calmar_ratio(returns: pd.Series, equity_curve: Optional[pd.Series] = None) -> float:
        """
        Calculate Calmar ratio (CAGR / Max Drawdown).

        Args:
            returns: Series of returns
            equity_curve: Optional equity curve (will be computed from returns if not provided)

        Returns:
            Calmar ratio
        """
        # Calculate CAGR
        geometric = PerformanceAnalytics.geometric_return(returns)

        # Calculate max drawdown
        if equity_curve is None:
            equity_curve = (1 + returns).cumprod()

        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = abs(drawdowns.min())

        if max_dd == 0:
            return 0.0

        calmar = geometric / max_dd

        return calmar

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Maximum drawdown (negative value)
        """
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        return drawdowns.min()

    @staticmethod
    def volatility(returns: pd.Series, periods: int = 252) -> float:
        """
        Calculate annualized volatility.

        Args:
            returns: Series of returns
            periods: Periods per year

        Returns:
            Annualized volatility
        """
        return returns.std() * np.sqrt(periods)


if __name__ == "__main__":
    # Test analytics
    print("\n" + "="*60)
    print("Testing Performance Analytics")
    print("="*60)

    # Create sample equity curve
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    np.random.seed(42)

    # Simulate returns with some drift and volatility
    returns = np.random.randn(1000) * 0.01 + 0.0003
    equity_curve = pd.Series(100000 * np.exp(returns.cumsum()), index=dates)

    # Create benchmark (lower return, lower vol)
    benchmark_returns = np.random.randn(1000) * 0.008 + 0.0002
    benchmark = pd.Series(100000 * np.exp(benchmark_returns.cumsum()), index=dates)

    print("\nEquity curve stats:")
    print(f"  Start: ${equity_curve.iloc[0]:,.2f}")
    print(f"  End: ${equity_curve.iloc[-1]:,.2f}")

    # Test 1: Basic metrics
    print("\n1. Calculating performance metrics...")
    metrics = PerformanceAnalytics.calculate_metrics(equity_curve)
    PerformanceAnalytics.print_report(metrics, "Strategy Performance")

    # Test 2: With benchmark
    print("\n2. Calculating metrics vs benchmark...")
    metrics_vs_benchmark = PerformanceAnalytics.calculate_metrics(
        equity_curve,
        benchmark=benchmark
    )
    PerformanceAnalytics.print_report(metrics_vs_benchmark, "Strategy vs Benchmark")

    # Test 3: Monthly returns
    print("\n3. Calculating monthly returns...")
    monthly_returns = PerformanceAnalytics.calculate_monthly_returns(equity_curve)
    print("\nMonthly returns table (first few rows):")
    print(monthly_returns.head())

    # Test 4: Drawdown periods
    print("\n4. Finding worst drawdown periods...")
    dd_periods = PerformanceAnalytics.calculate_drawdown_periods(equity_curve, top_n=3)
    print("\nTop 3 worst drawdowns:")
    print(dd_periods)

    # Test 5: Rolling metrics
    print("\n5. Calculating rolling metrics...")
    rolling = PerformanceAnalytics.calculate_rolling_metrics(
        equity_curve.pct_change().dropna(),
        window=252
    )
    print("\nRolling metrics (last 5 days):")
    print(rolling.tail())

    print("\n✅ Performance analytics test complete!")
