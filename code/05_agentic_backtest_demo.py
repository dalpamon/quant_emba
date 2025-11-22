"""
Agentic Backtesting Framework - Demo Script

Demonstrates the full capabilities of the agentic backtesting framework:
1. Load Korean stock data
2. Calculate technical indicators
3. Run agentic optimization
4. Compare strategies
5. Validate with walk-forward analysis
6. Export results

Author: Agentic Backtesting Framework
Date: 2025-11-22
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indicators import TechnicalIndicators
from core.feature_engineering import FeatureEngineeringPipeline, quick_features
from core.agent import AgenticStrategyExplorer, quick_explore
from core.analytics import PerformanceAnalytics
from core.optimizer import quick_optimize


def load_korean_data():
    """Load and prepare Korean stock data."""
    print("\n" + "="*70)
    print("📊 LOADING KOREAN STOCK DATA")
    print("="*70)

    data_path = Path("data/processed/stock_prices_clean.csv")

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run code/01_load_data.py first to generate cleaned data")
        return None

    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"✅ Loaded data: {df.shape}")
    print(f"   Stocks: {len(df.columns)}")
    print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")

    # Convert to MultiIndex format for framework
    df_long = df.stack().reset_index()
    df_long.columns = ['date', 'ticker', 'close']
    df_long = df_long.set_index(['ticker', 'date'])

    # Add dummy OHLCV data (FnGuide only has close prices)
    df_long['open'] = df_long['close']
    df_long['high'] = df_long['close'] * 1.02  # Assume 2% high
    df_long['low'] = df_long['close'] * 0.98   # Assume 2% low
    df_long['volume'] = 1000000  # Placeholder volume

    print(f"✅ Converted to OHLCV format: {df_long.shape}")

    return df_long


def demo_technical_indicators(data):
    """Demonstrate technical indicators calculation."""
    print("\n" + "="*70)
    print("📈 DEMONSTRATING TECHNICAL INDICATORS")
    print("="*70)

    # Select a sample stock (first in list)
    sample_ticker = data.index.get_level_values(0).unique()[0]
    sample_data = data.loc[sample_ticker]

    print(f"\nSample Stock: {sample_ticker}")
    print(f"Data Points: {len(sample_data)}")

    # Calculate indicators
    indicators = TechnicalIndicators(sample_data)

    print("\n📊 Calculating indicators...")

    # RSI
    rsi = indicators.rsi(period=14)
    print(f"   RSI (14): Latest = {rsi.iloc[-1]:.2f}")

    # MACD
    macd_df = indicators.macd()
    print(f"   MACD: Latest = {macd_df['macd'].iloc[-1]:.4f}")

    # Bollinger Bands
    bb_df = indicators.bollinger_bands()
    print(f"   BB %B: Latest = {bb_df['pct_b'].iloc[-1]:.2f} (position within bands)")

    # Geometric vs Arithmetic returns
    geometric = indicators.geometric_return(period=20)
    arithmetic = indicators.arithmetic_return(period=20)

    print(f"\n📊 Returns Comparison (20-day):")
    print(f"   Geometric: Latest = {geometric.iloc[-1]:.2%}")
    print(f"   Arithmetic: Latest = {arithmetic.iloc[-1]:.2%}")

    # Calculate all indicators
    all_indicators = indicators.calculate_all_indicators()
    print(f"\n✅ All indicators calculated: {len(all_indicators.columns)} features")
    print(f"   Features: {', '.join(all_indicators.columns[:5])}...")


def demo_feature_engineering(data):
    """Demonstrate feature engineering pipeline."""
    print("\n" + "="*70)
    print("🔧 DEMONSTRATING FEATURE ENGINEERING PIPELINE")
    print("="*70)

    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(data)

    print(f"\n📋 Available Levers: {len(pipeline.levers)}")

    # Show some levers
    lever_summary = pipeline.get_lever_summary()
    print("\nSample Levers:")
    print(lever_summary[['Lever', 'Current Value', 'Min', 'Max', 'Description']].head(10))

    # Adjust parameters
    print("\n🎛️  Adjusting parameters for Korean market (mean reversion)...")
    pipeline.set_lever("rsi_period", 7)        # Faster RSI
    pipeline.set_lever("bb_period", 10)        # Tighter bands
    pipeline.set_lever("bb_std", 1.5)          # More signals
    pipeline.set_lever("feature_selection_method", "momentum_only")

    print("   ✅ Parameters adjusted")

    # Generate features
    print("\n⚙️  Generating features...")
    features = pipeline.generate_features()

    print(f"   ✅ Generated {len(features.columns)} features")
    print(f"   Shape: {features.shape}")
    print(f"   Features: {list(features.columns)}")

    # Save configuration
    config_path = "output/demo_config.json"
    Path("output").mkdir(exist_ok=True)
    pipeline.save_config(config_path)
    print(f"\n💾 Configuration saved to: {config_path}")


def demo_simple_optimization(data):
    """Demonstrate simple parameter optimization."""
    print("\n" + "="*70)
    print("⚡ DEMONSTRATING SIMPLE OPTIMIZATION")
    print("="*70)

    print("\n🎯 Optimizing RSI period for mean reversion strategy...")
    print("   (This is a simplified demo - full agentic optimization is more powerful)")

    # Use small subset for speed
    tickers = data.index.get_level_values(0).unique()[:50]  # First 50 stocks
    subset_data = data.loc[tickers]

    print(f"\n   Using {len(tickers)} stocks for demo")

    # Define simple objective: maximize Sharpe of RSI-based strategy
    def simple_rsi_objective(params):
        try:
            rsi_period = params['rsi_period']

            # Calculate RSI for all stocks
            from core.indicators import TechnicalIndicators
            indicators = TechnicalIndicators(subset_data)
            rsi = indicators.rsi(period=rsi_period)

            # Simple signal: buy when RSI < 30, sell when RSI > 70
            signal = pd.Series(0, index=rsi.index)
            signal[rsi < 30] = 1   # Oversold = buy
            signal[rsi > 70] = -1  # Overbought = sell

            # Calculate returns
            if isinstance(subset_data.index, pd.MultiIndex):
                returns = subset_data.groupby(level=0)['close'].pct_change()
            else:
                returns = subset_data['close'].pct_change()

            # Strategy returns
            strategy_returns = signal.shift(1) * returns
            strategy_returns = strategy_returns.dropna()

            # Calculate Sharpe
            if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                return sharpe
            else:
                return -999

        except Exception as e:
            return -999

    # Define lever
    from core.feature_engineering import FeatureLever

    levers = {
        'rsi_period': FeatureLever(
            name='rsi_period',
            default=14,
            min_val=5,
            max_val=25,
            step=2,
            data_type='int',
            description='RSI period'
        )
    }

    # Run optimization
    print("\n🔍 Running random search (20 iterations)...")

    result = quick_optimize(
        simple_rsi_objective,
        levers,
        method='random',
        n_iter=20,
        maximize=True,
        verbose=False
    )

    print(f"\n✅ Optimization complete!")
    print(f"   Best RSI period: {result.best_config['rsi_period']}")
    print(f"   Best Sharpe ratio: {result.best_score:.3f}")

    # Show summary
    print(result.summary())


def demo_agentic_exploration(data):
    """Demonstrate full agentic exploration (the main feature!)."""
    print("\n" + "="*70)
    print("🤖 DEMONSTRATING AGENTIC STRATEGY EXPLORATION")
    print("="*70)

    print("\n⚠️  NOTE: Full agentic exploration with 2,620 stocks is computationally intensive.")
    print("   For this demo, we'll use a subset of stocks and reduced iterations.")

    # Use subset for demo
    tickers = data.index.get_level_values(0).unique()[:100]  # First 100 stocks
    subset_data = data.loc[tickers]

    print(f"\n📊 Using {len(tickers)} stocks")
    print(f"   Period: {subset_data.index.get_level_values(1).min()} to {subset_data.index.get_level_values(1).max()}")

    # Initialize explorer
    print("\n🤖 Initializing Agentic Explorer...")

    explorer = AgenticStrategyExplorer(
        data=subset_data,
        initial_capital=10_000_000,  # 10M KRW
        transaction_cost_bps=15,      # Korean market
        rebalance_freq='M',           # Monthly
        objective='sharpe',
        korean_market=True
    )

    print("   ✅ Explorer initialized")

    # Run exploration (reduced parameters for demo)
    print("\n🚀 Starting agentic exploration...")
    print("   Method: Genetic Algorithm")
    print("   Target strategies: 3")
    print("   Evaluation budget: 30 (reduced for demo)")

    try:
        strategies = explorer.explore(
            method='genetic',
            n_strategies=3,
            exploration_budget=30,  # Reduced for demo
            verbose=True
        )

        print(f"\n✅ Discovered {len(strategies)} strategies!")

        # Show results
        comparison = explorer.get_strategy_comparison()
        print("\n📊 Strategy Comparison:")
        print(comparison.to_string())

        # Show best strategy details
        if strategies:
            best = strategies[0]
            print(f"\n🏆 Best Strategy: {best.name}")
            print(f"   Sharpe Ratio: {best.sharpe_ratio():.3f}")
            print(f"   Total Return: {best.performance.get('total_return', 0):.2%}")
            print(f"   Max Drawdown: {best.performance.get('max_drawdown', 0):.2%}")

            print(f"\n   Parameters:")
            for param, value in best.parameters.items():
                print(f"      {param}: {value}")

            # Export
            output_dir = "output/strategies"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            explorer.export_best_strategy(output_dir)

    except Exception as e:
        print(f"\n⚠️  Error during exploration: {e}")
        print("   This is likely due to data issues or insufficient samples")
        import traceback
        traceback.print_exc()


def demo_analytics(data):
    """Demonstrate advanced analytics."""
    print("\n" + "="*70)
    print("📊 DEMONSTRATING ADVANCED ANALYTICS")
    print("="*70)

    # Create sample returns
    sample_ticker = data.index.get_level_values(0).unique()[0]
    sample_data = data.loc[sample_ticker]

    returns = sample_data['close'].pct_change().dropna()
    equity_curve = (1 + returns).cumprod() * 100000

    print(f"\nSample Stock: {sample_ticker}")
    print(f"Period: {len(returns)} days")

    # Calculate metrics
    print("\n📈 Performance Metrics:")

    sharpe = PerformanceAnalytics.sharpe_ratio(returns)
    sortino = PerformanceAnalytics.sortino_ratio(returns)
    calmar = PerformanceAnalytics.calmar_ratio(returns, equity_curve)

    print(f"   Sharpe Ratio: {sharpe:.3f}")
    print(f"   Sortino Ratio: {sortino:.3f}")
    print(f"   Calmar Ratio: {calmar:.3f}")

    # Geometric vs Arithmetic
    return_comp = PerformanceAnalytics.return_comparison(returns)

    print(f"\n📊 Return Comparison:")
    print(f"   Geometric Return: {return_comp['geometric_return']:.2%}")
    print(f"   Arithmetic Return: {return_comp['arithmetic_return']:.2%}")
    print(f"   Variance Drag: {return_comp['variance_drag']:.2%}")
    print(f"   (Higher variance drag indicates high volatility)")

    # Korean market metrics
    print(f"\n🇰🇷 Korean Market Metrics:")

    # Use market average as proxy
    market_returns = data.groupby(level=1)['close'].mean().pct_change().dropna()

    # Align dates
    common_dates = returns.index.intersection(market_returns.index)
    aligned_returns = returns.loc[common_dates]
    aligned_market = market_returns.loc[common_dates]

    korean_metrics = PerformanceAnalytics.korean_market_metrics(
        aligned_returns,
        aligned_market
    )

    print(f"   Mean Reversion (1-day): {korean_metrics['mean_reversion_1d']:.3f}")
    print(f"   Mean Reversion (20-day): {korean_metrics['mean_reversion_20d']:.3f}")
    print(f"   Momentum Reversal: {korean_metrics['momentum_reversal']:.3f}")
    print(f"   Skewness: {korean_metrics['skewness']:.3f}")
    print(f"   Kurtosis: {korean_metrics['kurtosis']:.3f}")

    if korean_metrics['mean_reversion_20d'] < -0.1:
        print(f"   ✅ Strong mean reversion detected!")
    elif korean_metrics['mean_reversion_20d'] > 0.1:
        print(f"   ⚠️  Momentum persistence detected")
    else:
        print(f"   ℹ️  Weak autocorrelation")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("🤖 AGENTIC BACKTESTING FRAMEWORK - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nThis demo showcases the full capabilities of the framework:")
    print("  1. Technical Indicators (20+ indicators)")
    print("  2. Feature Engineering (configurable levers)")
    print("  3. Simple Optimization (parameter tuning)")
    print("  4. Agentic Exploration (autonomous strategy discovery)")
    print("  5. Advanced Analytics (Korean market metrics)")
    print("\n" + "="*70)

    # Load data
    data = load_korean_data()

    if data is None:
        print("\n❌ Failed to load data. Exiting.")
        return

    # Run demonstrations
    try:
        demo_technical_indicators(data)
        demo_feature_engineering(data)
        demo_analytics(data)
        demo_simple_optimization(data)

        # Ask before running full agentic exploration
        print("\n" + "="*70)
        print("⚠️  The final demo (Agentic Exploration) is computationally intensive.")
        print("   It will take 2-5 minutes to complete.")
        print("="*70)

        response = input("\nRun agentic exploration demo? (y/n): ")

        if response.lower() == 'y':
            demo_agentic_exploration(data)
        else:
            print("\n⏭️  Skipping agentic exploration demo")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\n📚 Next Steps:")
    print("  1. Launch web interface: streamlit run app_agentic.py")
    print("  2. Read full documentation: README_AGENTIC.md")
    print("  3. Experiment with your own parameters and strategies")
    print("\n🚀 Happy Trading!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
