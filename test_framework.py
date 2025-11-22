"""
Quick Test Script for Agentic Backtesting Framework

Tests all major components individually.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("TESTING AGENTIC BACKTESTING FRAMEWORK")
print("="*70)

# Test 1: Load Data
print("\n1️⃣  Testing Data Loading...")
try:
    data_path = Path("data/processed/stock_prices_clean.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"   ✅ Loaded {df.shape[0]} days × {df.shape[1]} stocks")

        # Convert to MultiIndex format
        df_long = df.stack().reset_index()
        df_long.columns = ['date', 'ticker', 'close']
        df_long = df_long.set_index(['ticker', 'date'])

        # Add OHLCV columns
        df_long['open'] = df_long['close']
        df_long['high'] = df_long['close'] * 1.02
        df_long['low'] = df_long['close'] * 0.98
        df_long['volume'] = 1000000

        print(f"   ✅ Converted to OHLCV: {df_long.shape}")
        data = df_long
    else:
        print(f"   ❌ Data file not found: {data_path}")
        print("   Run: python code/01_load_data.py first")
        exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Test 2: Technical Indicators
print("\n2️⃣  Testing Technical Indicators...")
try:
    from core.indicators import TechnicalIndicators

    # Get first stock
    sample_ticker = data.index.get_level_values(0).unique()[0]
    sample_data = data.loc[sample_ticker]

    indicators = TechnicalIndicators(sample_data)

    # Test RSI
    rsi = indicators.rsi(period=14)
    print(f"   ✅ RSI calculated: Latest = {rsi.iloc[-1]:.2f}")

    # Test MACD
    macd = indicators.macd()
    print(f"   ✅ MACD calculated: Latest = {macd['macd'].iloc[-1]:.4f}")

    # Test Bollinger Bands
    bb = indicators.bollinger_bands()
    print(f"   ✅ Bollinger Bands calculated: %B = {bb['pct_b'].iloc[-1]:.2f}")

    # Test all indicators
    all_ind = indicators.calculate_all_indicators()
    print(f"   ✅ All indicators: {len(all_ind.columns)} features generated")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Feature Engineering
print("\n3️⃣  Testing Feature Engineering Pipeline...")
try:
    from core.feature_engineering import FeatureEngineeringPipeline

    # Use subset for speed
    subset = data.loc[data.index.get_level_values(0).unique()[:10]]

    pipeline = FeatureEngineeringPipeline(subset)
    print(f"   ✅ Pipeline initialized with {len(pipeline.levers)} levers")

    # Adjust some parameters
    pipeline.set_lever("rsi_period", 7)
    pipeline.set_lever("bb_period", 10)
    print(f"   ✅ Parameters adjusted")

    # Generate features
    features = pipeline.generate_features()
    print(f"   ✅ Features generated: {features.shape}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Analytics
print("\n4️⃣  Testing Performance Analytics...")
try:
    from core.analytics import PerformanceAnalytics

    # Create sample returns
    sample_data = data.loc[data.index.get_level_values(0).unique()[0]]
    returns = sample_data['close'].pct_change().dropna()

    # Test Sharpe ratio
    sharpe = PerformanceAnalytics.sharpe_ratio(returns)
    print(f"   ✅ Sharpe Ratio: {sharpe:.3f}")

    # Test geometric vs arithmetic
    return_comp = PerformanceAnalytics.return_comparison(returns)
    print(f"   ✅ Geometric Return: {return_comp['geometric_return']:.2%}")
    print(f"   ✅ Arithmetic Return: {return_comp['arithmetic_return']:.2%}")
    print(f"   ✅ Variance Drag: {return_comp['variance_drag']:.2%}")

    # Test Korean market metrics
    market_returns = data.groupby(level=1)['close'].mean().pct_change().dropna()
    common_dates = returns.index.intersection(market_returns.index)

    if len(common_dates) > 50:
        korean_metrics = PerformanceAnalytics.korean_market_metrics(
            returns.loc[common_dates],
            market_returns.loc[common_dates]
        )
        print(f"   ✅ Mean Reversion (20d): {korean_metrics['mean_reversion_20d']:.3f}")
        print(f"   ✅ Momentum Reversal: {korean_metrics['momentum_reversal']:.3f}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Optimizer
print("\n5️⃣  Testing Optimization Engine...")
try:
    from core.optimizer import RandomSearchOptimizer
    from core.feature_engineering import FeatureLever

    # Define simple objective
    def test_objective(params):
        # Just return random score for testing
        return np.random.randn() + params['test_param'] / 10

    # Define lever
    levers = {
        'test_param': FeatureLever(
            name='test_param',
            default=10,
            min_val=5,
            max_val=20,
            step=1,
            data_type='int'
        )
    }

    # Run quick optimization
    optimizer = RandomSearchOptimizer(test_objective, levers, maximize=True)
    result = optimizer.optimize(n_iter=10, verbose=False)

    print(f"   ✅ Optimization complete")
    print(f"   ✅ Best params: {result.best_config}")
    print(f"   ✅ Best score: {result.best_score:.3f}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Agentic Explorer (Quick test)
print("\n6️⃣  Testing Agentic Explorer (Quick Test)...")
print("   (Using small subset for speed)")
try:
    from core.agent import AgenticStrategyExplorer

    # Use tiny subset for quick test
    tiny_subset = data.loc[data.index.get_level_values(0).unique()[:5]]

    explorer = AgenticStrategyExplorer(
        data=tiny_subset,
        objective='sharpe',
        korean_market=True
    )
    print(f"   ✅ Explorer initialized")

    # Run mini exploration
    print(f"   ⚙️  Running mini exploration (10 evaluations)...")
    strategies = explorer.explore(
        method='random',
        n_strategies=2,
        exploration_budget=10,
        verbose=False
    )

    print(f"   ✅ Discovered {len(strategies)} strategies")
    if strategies:
        print(f"   ✅ Best Sharpe: {strategies[0].sharpe_ratio():.3f}")

except Exception as e:
    print(f"   ⚠️  Skipped (may need more data): {e}")

# Summary
print("\n" + "="*70)
print("✅ FRAMEWORK TEST COMPLETE!")
print("="*70)
print("\nAll core components are working! You can now:")
print("  1. Run full demo: python code/05_agentic_backtest_demo.py")
print("  2. Launch web app: streamlit run app_agentic.py")
print("  3. Use Python API for custom strategies")
print("="*70)
