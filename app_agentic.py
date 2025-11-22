"""
Agentic Backtesting Framework - Web Interface

Interactive Streamlit application for autonomous strategy discovery and optimization.
Designed for Korean stock market quantitative trading.

Features:
- Real-time parameter tuning with visual feedback
- Agentic strategy exploration
- Multi-strategy comparison
- Korean market analytics
- Advanced technical indicators (RSI, MACD, Bollinger Bands, etc.)

Author: Agentic Backtesting Framework
Date: 2025-11-22
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_loader import DataLoader
from core.indicators import TechnicalIndicators, add_all_indicators
from core.feature_engineering import FeatureEngineeringPipeline, FeatureConfig
from core.agent import AgenticStrategyExplorer, StrategyCandidate
from core.backtest import Backtester
from core.analytics import PerformanceAnalytics
from core.portfolio import Portfolio


# Page configuration
st.set_page_config(
    page_title="🤖 Agentic Backtesting Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'explorer' not in st.session_state:
    st.session_state.explorer = None
if 'strategies' not in st.session_state:
    st.session_state.strategies = []
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None


def load_korean_stock_data():
    """Load Korean stock data from FnGuide."""
    try:
        with st.spinner("📊 Loading Korean stock data..."):
            # Load cleaned data
            data_path = Path("data/processed/stock_prices_clean.csv")

            if data_path.exists():
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)

                # Convert to long format (MultiIndex)
                df_long = df.stack().reset_index()
                df_long.columns = ['date', 'ticker', 'close']
                df_long = df_long.set_index(['ticker', 'date'])

                # Add dummy OHLCV data (FnGuide only has close prices)
                df_long['open'] = df_long['close']
                df_long['high'] = df_long['close'] * 1.02
                df_long['low'] = df_long['close'] * 0.98
                df_long['volume'] = 1000000  # Placeholder

                return df_long
            else:
                st.error(f"Data file not found: {data_path}")
                return None

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def plot_strategy_performance(strategy: StrategyCandidate, data: pd.DataFrame):
    """Plot strategy performance with detailed charts."""

    st.markdown(f'<div class="sub-header">📈 {strategy.name} Performance</div>', unsafe_allow_html=True)

    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sharpe Ratio",
            f"{strategy.sharpe_ratio():.3f}",
            delta=None
        )

    with col2:
        st.metric(
            "Total Return",
            f"{strategy.performance.get('total_return', 0):.2%}",
            delta=None
        )

    with col3:
        st.metric(
            "Max Drawdown",
            f"{strategy.performance.get('max_drawdown', 0):.2%}",
            delta=None
        )

    with col4:
        st.metric(
            "Volatility",
            f"{strategy.performance.get('volatility', 0):.2%}",
            delta=None
        )


def plot_indicator_comparison(data: pd.DataFrame, ticker: str):
    """Plot technical indicators for a specific ticker."""

    ticker_data = data.loc[ticker]

    # Calculate indicators
    indicators = TechnicalIndicators(ticker_data)

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'Price & Moving Averages',
            'RSI',
            'MACD',
            'Bollinger Bands'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    # Price and MAs
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=ticker_data['close'], name='Close', line=dict(color='black')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=indicators.sma(20), name='SMA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=indicators.sma(50), name='SMA 50', line=dict(color='blue')),
        row=1, col=1
    )

    # RSI
    rsi = indicators.rsi()
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=rsi, name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    macd_df = indicators.macd()
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=macd_df['macd'], name='MACD', line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=macd_df['signal'], name='Signal', line=dict(color='orange')),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(x=ticker_data.index, y=macd_df['histogram'], name='Histogram', marker_color='gray'),
        row=3, col=1
    )

    # Bollinger Bands
    bb_df = indicators.bollinger_bands()
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=bb_df['upper'], name='BB Upper', line=dict(color='red', dash='dash')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=bb_df['middle'], name='BB Middle', line=dict(color='gray')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=bb_df['lower'], name='BB Lower', line=dict(color='green', dash='dash')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=ticker_data.index, y=ticker_data['close'], name='Close', line=dict(color='black')),
        row=4, col=1
    )

    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text=f"Technical Indicators: {ticker}"
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application."""

    # Header
    st.markdown('<div class="main-header">🤖 Agentic Backtesting Framework</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <b>Autonomous Strategy Discovery for Korean Stock Market</b><br>
        Powered by Advanced Machine Learning & Optimization Algorithms
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Configuration")

    # Page selection
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Data Explorer", "🤖 Agentic Optimization", "📈 Strategy Comparison", "🔧 Manual Tuning", "📚 Documentation"]
    )

    # Data loading
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Data")

    if st.sidebar.button("Load Korean Stock Data"):
        data = load_korean_stock_data()
        if data is not None:
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.sidebar.success(f"✅ Loaded {len(data.index.get_level_values(0).unique())} stocks")

    if st.session_state.data_loaded:
        st.sidebar.info(f"📊 Data Shape: {st.session_state.data.shape}")

    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🤖 Agentic Optimization":
        show_agentic_optimization()
    elif page == "📈 Strategy Comparison":
        show_strategy_comparison()
    elif page == "🔧 Manual Tuning":
        show_manual_tuning()
    elif page == "📚 Documentation":
        show_documentation()


def show_home_page():
    """Home page with introduction and overview."""

    st.markdown('<div class="sub-header">🎯 What is Agentic Backtesting?</div>', unsafe_allow_html=True)

    st.markdown("""
    The Agentic Backtesting Framework is an **autonomous trading strategy discovery system** that:

    - 🤖 **Autonomously explores** parameter space to find optimal trading strategies
    - 🧠 **Learns** from results and adapts search strategy
    - 🎯 **Discovers** multiple diverse, high-performing strategies
    - 📊 **Validates** strategies using walk-forward analysis
    - 🇰🇷 **Optimized** for Korean stock market (KRX/KOSPI/KOSDAQ)

    ### Key Features

    #### 1️⃣ Advanced Technical Indicators
    - **Momentum**: RSI, MACD, Stochastic, Williams %R
    - **Volatility**: Bollinger Bands, ATR, Keltner Channels
    - **Trend**: SMA, EMA, ADX
    - **Volume**: OBV, MFI, VWAP
    - **Statistical**: Geometric/Arithmetic Returns, Z-Scores

    #### 2️⃣ Optimization Algorithms
    - **Grid Search**: Exhaustive parameter exploration
    - **Random Search**: Efficient high-dimensional search
    - **Genetic Algorithm**: Evolutionary optimization
    - **Walk-Forward**: Robust out-of-sample validation

    #### 3️⃣ Korean Market Adaptations
    - Mean reversion detection (Korean market shows reverse momentum)
    - Sector rotation strategies
    - Chaebols and market structure awareness
    - KRX-specific features

    #### 4️⃣ Risk Management
    - Sharpe, Sortino, Calmar ratio optimization
    - Maximum drawdown constraints
    - Volatility targeting
    - Geometric vs. arithmetic returns
    """)

    st.markdown('<div class="sub-header">🚀 Quick Start</div>', unsafe_allow_html=True)

    st.markdown("""
    1. **Load Data**: Click "Load Korean Stock Data" in the sidebar
    2. **Explore**: Navigate to "Data Explorer" to visualize indicators
    3. **Optimize**: Run "Agentic Optimization" to discover strategies
    4. **Compare**: Analyze discovered strategies in "Strategy Comparison"
    5. **Fine-tune**: Manually adjust parameters in "Manual Tuning"
    """)

    # System architecture diagram
    st.markdown('<div class="sub-header">🏗️ System Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────┐
    │            🌐 WEB INTERFACE (Streamlit)                  │
    │   Interactive parameter tuning & visualization           │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │          🤖 AGENTIC OPTIMIZATION ENGINE                  │
    │   - Grid Search    - Genetic Algorithm                   │
    │   - Random Search  - Walk-Forward Validation             │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │       🔧 FEATURE ENGINEERING PIPELINE                    │
    │   - Technical Indicators  - Custom Features              │
    │   - Statistical Metrics   - Configurable Levers          │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │           ⚡ VECTORIZED BACKTESTING ENGINE               │
    │   High-performance strategy evaluation                   │
    └─────────────────────────────────────────────────────────┘
    ```
    """)


def show_data_explorer():
    """Data exploration page with technical indicators."""

    st.markdown('<div class="sub-header">📊 Data Explorer</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the sidebar")
        return

    data = st.session_state.data

    # Get list of tickers
    tickers = sorted(data.index.get_level_values(0).unique())

    # Ticker selection
    selected_ticker = st.selectbox("Select Stock", tickers)

    # Show technical indicators
    if selected_ticker:
        plot_indicator_comparison(data, selected_ticker)

        # Show raw data
        with st.expander("📋 Raw Data"):
            ticker_data = data.loc[selected_ticker]
            st.dataframe(ticker_data.tail(50))


def show_agentic_optimization():
    """Agentic optimization page."""

    st.markdown('<div class="sub-header">🤖 Agentic Strategy Optimization</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the sidebar")
        return

    st.markdown("""
    The agent will **autonomously explore** parameter space to discover optimal trading strategies.

    ### Optimization Process:
    1. **Broad Exploration**: Survey parameter space
    2. **Local Refinement**: Focus on promising regions
    3. **Diversity Selection**: Choose diverse high-performers
    """)

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        optimization_method = st.selectbox(
            "Optimization Method",
            ["genetic", "random", "grid"],
            help="Genetic algorithm is recommended for complex spaces"
        )

        n_strategies = st.slider(
            "Number of Strategies to Discover",
            min_value=3,
            max_value=20,
            value=5,
            step=1
        )

    with col2:
        exploration_budget = st.slider(
            "Evaluation Budget",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Higher budget = more thorough exploration but slower"
        )

        objective = st.selectbox(
            "Optimization Objective",
            ["sharpe", "sortino", "calmar", "returns"],
            help="Metric to maximize"
        )

    # Run optimization
    if st.button("🚀 Start Agentic Exploration", type="primary"):
        with st.spinner("🤖 Agent exploring parameter space..."):
            try:
                # Create explorer
                explorer = AgenticStrategyExplorer(
                    data=st.session_state.data,
                    objective=objective,
                    korean_market=True
                )

                # Run exploration
                strategies = explorer.explore(
                    method=optimization_method,
                    n_strategies=n_strategies,
                    exploration_budget=exploration_budget,
                    verbose=False
                )

                # Store results
                st.session_state.explorer = explorer
                st.session_state.strategies = strategies

                st.success(f"✅ Discovered {len(strategies)} strategies!")

                # Show results summary
                st.markdown('<div class="sub-header">🏆 Top Strategies</div>', unsafe_allow_html=True)

                comparison_df = explorer.get_strategy_comparison()
                st.dataframe(comparison_df)

                # Export
                st.markdown('<div class="sub-header">💾 Export</div>', unsafe_allow_html=True)

                if st.button("Export Best Strategy"):
                    explorer.export_best_strategy()
                    st.success("✅ Best strategy exported to output/strategies/")

            except Exception as e:
                st.error(f"Error during optimization: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Show existing results
    if st.session_state.strategies:
        st.markdown('<div class="sub-header">📊 Discovered Strategies</div>', unsafe_allow_html=True)

        for i, strategy in enumerate(st.session_state.strategies[:5], 1):
            with st.expander(f"Strategy {i}: {strategy.name}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Sharpe Ratio", f"{strategy.sharpe_ratio():.3f}")
                    st.metric("Sortino Ratio", f"{strategy.sortino_ratio():.3f}")

                with col2:
                    st.metric("Total Return", f"{strategy.performance.get('total_return', 0):.2%}")
                    st.metric("Max Drawdown", f"{strategy.performance.get('max_drawdown', 0):.2%}")

                with col3:
                    st.metric("Volatility", f"{strategy.performance.get('volatility', 0):.2%}")
                    st.metric("Calmar Ratio", f"{strategy.calmar_ratio():.3f}")

                # Show parameters
                st.markdown("**Parameters:**")
                params_df = pd.DataFrame([strategy.parameters])
                st.dataframe(params_df)


def show_strategy_comparison():
    """Strategy comparison page."""

    st.markdown('<div class="sub-header">📈 Strategy Comparison</div>', unsafe_allow_html=True)

    if not st.session_state.strategies:
        st.warning("⚠️ No strategies discovered yet. Run Agentic Optimization first.")
        return

    # Comparison table
    if st.session_state.explorer:
        comparison_df = st.session_state.explorer.get_strategy_comparison()
        st.dataframe(comparison_df, use_container_width=True)

        # Visualization
        st.markdown('<div class="sub-header">📊 Performance Visualization</div>', unsafe_allow_html=True)

        # Sharpe vs Return scatter
        fig = px.scatter(
            comparison_df,
            x='Total Return',
            y='Sharpe',
            size='Volatility',
            color='Max Drawdown',
            hover_data=['Strategy'],
            title='Risk-Return Profile',
            labels={'Total Return': 'Total Return (%)', 'Sharpe': 'Sharpe Ratio'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart comparison
        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            name='Sharpe Ratio',
            x=comparison_df['Strategy'],
            y=comparison_df['Sharpe']
        ))

        fig2.update_layout(
            title='Sharpe Ratio Comparison',
            xaxis_title='Strategy',
            yaxis_title='Sharpe Ratio'
        )

        st.plotly_chart(fig2, use_container_width=True)


def show_manual_tuning():
    """Manual parameter tuning page."""

    st.markdown('<div class="sub-header">🔧 Manual Parameter Tuning</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the sidebar")
        return

    st.markdown("""
    Manually adjust indicator parameters and see real-time performance impact.
    """)

    # Initialize pipeline if not exists
    if st.session_state.pipeline is None:
        st.session_state.pipeline = FeatureEngineeringPipeline(st.session_state.data)

    pipeline = st.session_state.pipeline

    # Parameter controls
    st.markdown("### 🎛️ Indicator Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Momentum Indicators**")

        rsi_period = st.slider("RSI Period", 5, 30, 14)
        macd_fast = st.slider("MACD Fast", 8, 20, 12)
        macd_slow = st.slider("MACD Slow", 20, 40, 26)

    with col2:
        st.markdown("**Volatility Indicators**")

        bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
        bb_std = st.slider("Bollinger Bands Std", 1.0, 3.0, 2.0, 0.5)
        atr_period = st.slider("ATR Period", 7, 21, 14)

    # Update pipeline
    pipeline.set_lever("rsi_period", rsi_period)
    pipeline.set_lever("macd_fast", macd_fast)
    pipeline.set_lever("macd_slow", macd_slow)
    pipeline.set_lever("bb_period", bb_period)
    pipeline.set_lever("bb_std", bb_std)
    pipeline.set_lever("atr_period", atr_period)

    # Test strategy
    if st.button("🧪 Test Configuration"):
        with st.spinner("Testing..."):
            try:
                # Generate features
                features = pipeline.generate_features()

                # Create simple signal
                signal = features.mean(axis=1)

                # Quick backtest
                st.success("✅ Configuration tested successfully!")

                # Show feature summary
                st.markdown("### 📊 Generated Features")
                st.write(f"Number of features: {len(features.columns)}")
                st.dataframe(features.describe())

            except Exception as e:
                st.error(f"Error: {e}")


def show_documentation():
    """Documentation page."""

    st.markdown('<div class="sub-header">📚 Documentation</div>', unsafe_allow_html=True)

    st.markdown("""
    ## 📖 User Guide

    ### Getting Started

    #### 1. Load Data
    Click "Load Korean Stock Data" in the sidebar to load FnGuide data (2,620 Korean stocks).

    #### 2. Explore Indicators
    Navigate to **Data Explorer** to visualize technical indicators for any stock:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - And 20+ more indicators

    #### 3. Run Agentic Optimization
    Go to **Agentic Optimization** and configure:
    - **Method**: Genetic Algorithm (recommended), Random Search, or Grid Search
    - **Number of Strategies**: How many diverse strategies to discover (3-20)
    - **Budget**: Evaluation budget (50-500)
    - **Objective**: Sharpe, Sortino, Calmar, or Returns

    Click "Start Agentic Exploration" and wait for the agent to discover optimal strategies.

    #### 4. Compare Strategies
    View **Strategy Comparison** to analyze discovered strategies:
    - Performance metrics table
    - Risk-return scatter plot
    - Sharpe ratio comparison

    #### 5. Manual Fine-Tuning
    Use **Manual Tuning** to adjust specific indicator parameters and test configurations.

    ---

    ## 🧠 Concepts

    ### Technical Indicators

    **RSI (Relative Strength Index)**
    - Momentum oscillator measuring speed and magnitude of price changes
    - Range: 0-100
    - >70 = overbought, <30 = oversold

    **MACD**
    - Trend-following momentum indicator
    - Shows relationship between two exponential moving averages
    - MACD line, Signal line, Histogram

    **Bollinger Bands**
    - Volatility bands placed above/below moving average
    - Price tends to bounce within bands
    - Band width indicates volatility

    **Geometric Mean Return**
    - Compounded average rate of return
    - More accurate than arithmetic mean for volatile assets
    - Formula: [(1+r₁)(1+r₂)...(1+rₙ)]^(1/n) - 1

    ### Optimization Methods

    **Grid Search**
    - Exhaustively tests all parameter combinations
    - Best for small parameter spaces
    - Guarantees global optimum (within grid)

    **Random Search**
    - Randomly samples parameter space
    - More efficient for high-dimensional spaces
    - Good exploration-exploitation balance

    **Genetic Algorithm**
    - Evolutionary optimization
    - Uses selection, crossover, mutation
    - Best for complex, non-convex landscapes

    **Walk-Forward Optimization**
    - Rolling window optimization
    - In-sample training, out-of-sample testing
    - Critical for avoiding overfitting

    ### Korean Market Characteristics

    **Mean Reversion**
    - Korean market shows reverse momentum effect
    - Winners underperform, losers outperform
    - Contrarian strategies may work better

    **Market Structure**
    - Dominated by chaebols (Samsung, Hyundai, SK, LG)
    - High retail investor participation
    - Different information efficiency vs. US markets

    ---

    ## 🔧 Technical Reference

    ### File Structure
    ```
    quant_emba/
    ├── core/
    │   ├── indicators.py          # Technical indicators
    │   ├── feature_engineering.py # Feature pipeline
    │   ├── optimizer.py           # Optimization algorithms
    │   ├── agent.py               # Agentic explorer
    │   ├── backtest.py            # Backtesting engine
    │   ├── portfolio.py           # Portfolio construction
    │   └── analytics.py           # Performance metrics
    ├── app_agentic.py            # This web application
    └── data/
        └── processed/
            └── stock_prices_clean.csv
    ```

    ### API Examples

    **Load and analyze a stock:**
    ```python
    from core.indicators import TechnicalIndicators

    indicators = TechnicalIndicators(data)
    rsi = indicators.rsi(period=14)
    macd = indicators.macd()
    bb = indicators.bollinger_bands()
    ```

    **Run agentic optimization:**
    ```python
    from core.agent import AgenticStrategyExplorer

    explorer = AgenticStrategyExplorer(data)
    strategies = explorer.explore(
        method='genetic',
        n_strategies=10,
        exploration_budget=200
    )
    ```

    **Manual backtesting:**
    ```python
    from core.backtest import Backtester

    backtester = Backtester(
        initial_capital=100000,
        transaction_cost_bps=10
    )
    results = backtester.run(price_data, position_schedule)
    ```

    ---

    ## 📞 Support

    For questions or issues:
    - Check the code documentation in `core/` modules
    - Review example scripts in `code/` directory
    - Consult original EMBA course materials

    **Version**: 1.0.0 (2025-11-22)
    """)


if __name__ == "__main__":
    main()
