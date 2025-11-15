"""
Factor Lab - Streamlit Web Application
Interactive backtesting platform for quantitative strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

from core import (
    DatabaseSchema,
    DataLoader,
    FactorEngine,
    Portfolio,
    PortfolioHistory,
    Backtester,
    PerformanceAnalytics
)


# Page configuration
st.set_page_config(
    page_title="Factor Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #00D9FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'strategy_config' not in st.session_state:
    st.session_state.strategy_config = None


def main():
    """Main application"""

    # Sidebar Navigation
    st.sidebar.title("🧪 Factor Lab")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Strategy Builder", "Results", "Factor Explorer"]
    )

    if page == "Home":
        show_home_page()
    elif page == "Strategy Builder":
        show_strategy_builder()
    elif page == "Results":
        show_results_page()
    elif page == "Factor Explorer":
        show_factor_explorer()


def show_home_page():
    """Homepage with introduction and quick start"""

    st.markdown('<h1 class="main-header">📊 Factor Lab</h1>', unsafe_allow_html=True)
    st.markdown("### Build & Test Quantitative Investment Strategies")

    # Hero section
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**1. Select Factors**\nChoose momentum, value, quality, and more")

    with col2:
        st.info("**2. Build Strategy**\nCombine factors with custom weights")

    with col3:
        st.info("**3. Analyze Results**\nView performance metrics and charts")

    st.markdown("---")

    # Quick start
    st.subheader("🚀 Quick Start")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Try Example Strategy", type="primary", use_container_width=True):
            # Pre-load an example strategy
            st.session_state.strategy_config = {
                'universe': 'tech',
                'factors': {'momentum_12m': 0.5, 'value_pb': 0.5},
                'start_date': '2020-01-01',
                'end_date': '2024-10-25'
            }
            st.success("✅ Example strategy loaded! Go to Strategy Builder to run it.")

    with col2:
        if st.button("Build Your Own", use_container_width=True):
            st.info("👈 Go to Strategy Builder in the sidebar")

    # Educational content
    with st.expander("💡 What is Factor Investing?"):
        st.write("""
        Factor investing is a strategy that targets specific drivers of return
        across asset classes. Common factors include:

        - **Momentum**: Stocks that have performed well continue to perform well
        - **Value**: Cheap stocks (low P/B, P/E) outperform
        - **Size**: Small-cap stocks earn higher returns
        - **Quality**: Profitable companies with stable earnings
        - **Low Volatility**: Lower-risk stocks earn comparable returns

        **Academic Foundation:**
        - Fama & French (1993) - Three Factor Model
        - Jegadeesh & Titman (1993) - Momentum
        - Novy-Marx (2013) - Quality Factor
        """)

    st.markdown("---")

    # Features
    st.subheader("✨ Features")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**📊 5 Quantitative Factors**")
        st.write("• Momentum (12M, 6M, 3M)")
        st.write("• Value (P/B, P/E, P/S)")
        st.write("• Quality (ROE, ROA, Margin)")
        st.write("• Size (Market Cap)")
        st.write("• Low Volatility (60-day)")

    with col2:
        st.write("**📈 Advanced Analytics**")
        st.write("• Sharpe, Sortino, Calmar Ratios")
        st.write("• Maximum Drawdown Analysis")
        st.write("• Transaction Cost Modeling")
        st.write("• Interactive Charts (Plotly)")

    st.markdown("---")
    st.caption("Built for SNU EMBA - Inefficient Market and Quant Investment")


def show_strategy_builder():
    """Strategy Builder page"""

    st.title("🔧 Strategy Builder")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Universe selection
    loader = DataLoader('quant1_data.db')
    universes = loader.get_universe_info()

    selected_universe = st.sidebar.selectbox(
        "Stock Universe",
        options=list(universes.keys()),
        format_func=lambda x: universes[x]
    )

    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2020, 1, 1),
            min_value=datetime(2015, 1, 1),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now() - timedelta(days=1),
            min_value=datetime(2015, 1, 1),
            max_value=datetime.now()
        )

    st.sidebar.markdown("---")

    # Factor weights
    st.sidebar.header("📊 Factor Weights")

    st.sidebar.caption("Adjust factor weights (will auto-normalize to 100%)")

    momentum_weight = st.sidebar.slider(
        "Momentum (12M)",
        0, 100, 40,
        help="Stocks with strong past performance"
    )

    value_weight = st.sidebar.slider(
        "Value (P/B)",
        0, 100, 30,
        help="Cheap stocks relative to book value"
    )

    quality_weight = st.sidebar.slider(
        "Quality (ROE)",
        0, 100, 30,
        help="Profitable companies"
    )

    size_weight = st.sidebar.slider(
        "Size (Market Cap)",
        0, 100, 0,
        help="Small-cap premium"
    )

    volatility_weight = st.sidebar.slider(
        "Low Volatility",
        0, 100, 0,
        help="Lower-risk stocks"
    )

    # Validate weights
    total_weight = momentum_weight + value_weight + quality_weight + size_weight + volatility_weight

    if total_weight == 0:
        st.sidebar.error("⚠️ Please select at least one factor")
        return

    if total_weight != 100:
        st.sidebar.info(f"ℹ️ Total: {total_weight}% (will normalize to 100%)")

    # Normalize weights
    factor_weights = {
        'momentum_12m': momentum_weight / total_weight,
        'value_pb': value_weight / total_weight,
        'quality_roe': quality_weight / total_weight,
        'size_log_mcap': size_weight / total_weight,
        'volatility_60d': volatility_weight / total_weight
    }

    # Remove zero-weight factors
    factor_weights = {k: v for k, v in factor_weights.items() if v > 0}

    st.sidebar.markdown("---")

    # Portfolio settings
    st.sidebar.header("⚙️ Portfolio Settings")

    portfolio_type = st.sidebar.selectbox(
        "Strategy Type",
        ["Long-Short", "Long-Only"]
    )

    rebalance_freq = st.sidebar.selectbox(
        "Rebalancing",
        ["Monthly", "Quarterly", "Annually"]
    )

    transaction_cost = st.sidebar.number_input(
        "Transaction Cost (bps)",
        min_value=0,
        max_value=100,
        value=10,
        help="Basis points per trade (10 bps = 0.1%)"
    )

    # Main area - Strategy Summary
    st.subheader("Strategy Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Universe", universes[selected_universe])
        st.metric("Period", f"{(end_date - start_date).days} days")

    with col2:
        st.metric("Active Factors", len(factor_weights))
        st.metric("Rebalancing", rebalance_freq)

    with col3:
        st.metric("Strategy Type", portfolio_type)
        st.metric("Transaction Cost", f"{transaction_cost} bps")

    st.markdown("---")

    # Display factor weights
    st.subheader("Factor Allocation")

    if factor_weights:
        weights_df = pd.DataFrame(
            list(factor_weights.items()),
            columns=['Factor', 'Weight']
        )
        weights_df['Weight %'] = weights_df['Weight'].apply(lambda x: f"{x*100:.1f}%")

        st.dataframe(
            weights_df[['Factor', 'Weight %']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("⚠️ Please select at least one factor")

    st.markdown("---")

    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):

        if not factor_weights:
            st.error("❌ Please select at least one factor")
            return

        try:
            with st.spinner("Running backtest... This may take 30-60 seconds"):

                # Progress tracking
                progress_bar = st.progress(0, text="Initializing...")

                # Step 1: Load data
                progress_bar.progress(10, text="Loading stock universe...")
                tickers = loader.get_universe_tickers(selected_universe)

                progress_bar.progress(20, text=f"Downloading price data for {len(tickers)} stocks...")
                prices = loader.fetch_prices(
                    tickers,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

                if prices.empty:
                    st.error("❌ Failed to download price data. Please try again.")
                    return

                progress_bar.progress(40, text="Fetching fundamental data...")
                fundamentals = loader.fetch_fundamentals(tickers)

                # Step 2: Calculate factors
                progress_bar.progress(50, text="Calculating factor scores...")
                factors = FactorEngine.calculate_all_factors(prices, fundamentals, tickers)

                progress_bar.progress(60, text="Normalizing factors...")
                normalized_factors = FactorEngine.normalize_factors(factors)

                progress_bar.progress(65, text="Combining factors...")
                composite_scores = FactorEngine.combine_factors(normalized_factors, factor_weights)

                # Step 3: Build portfolio
                progress_bar.progress(70, text="Constructing portfolio...")

                # Determine portfolio type
                portfolio_long_pct = 0.2
                portfolio_short_pct = 0.2 if portfolio_type == "Long-Short" else 0.0

                portfolio_builder = Portfolio(
                    long_pct=portfolio_long_pct,
                    short_pct=portfolio_short_pct,
                    portfolio_type=portfolio_type.lower()
                )

                # Get rebalancing dates
                freq_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Annually': 'Y'}

                # Extract close prices for rebalancing
                if isinstance(prices.columns, pd.MultiIndex):
                    close_prices = pd.DataFrame()
                    for ticker in tickers:
                        if ticker in prices.columns.get_level_values(0):
                            close_prices[ticker] = prices[ticker]['Close']
                else:
                    close_prices = prices

                rebal_dates = close_prices.resample(freq_map[rebalance_freq]).last().index

                # Build positions for each rebalancing date
                positions_by_date = {}
                weights_by_date = {}

                progress_bar.progress(75, text="Building positions for each rebalancing...")

                for date in rebal_dates:
                    date_str = date.strftime('%Y-%m-%d')
                    if date in close_prices.index:
                        positions = portfolio_builder.construct_portfolio(composite_scores)
                        weights = portfolio_builder.get_weights(positions, weighting='equal')

                        positions_by_date[date_str] = positions
                        weights_by_date[date_str] = weights

                # Step 4: Run backtest
                progress_bar.progress(85, text="Running backtest simulation...")

                backtester = Backtester(
                    initial_capital=100000,
                    transaction_cost_bps=transaction_cost,
                    rebalance_freq=freq_map[rebalance_freq]
                )

                results = backtester.run(close_prices, positions_by_date, weights_by_date)

                # Step 5: Calculate metrics
                progress_bar.progress(95, text="Calculating performance metrics...")

                metrics = PerformanceAnalytics.calculate_metrics(results['portfolio_value'])

                # Store in session state
                st.session_state.backtest_results = {
                    'equity_curve': results,
                    'metrics': metrics,
                    'positions': positions_by_date,
                    'weights': weights_by_date,
                    'strategy_config': {
                        'factors': factor_weights,
                        'universe': selected_universe,
                        'dates': (start_date, end_date),
                        'rebalance': rebalance_freq,
                        'portfolio_type': portfolio_type,
                        'transaction_cost': transaction_cost
                    }
                }

                progress_bar.progress(100, text="Complete!")
                progress_bar.empty()

                st.success("✅ Backtest complete! View results in the Results page.")

                # Show quick summary
                st.subheader("Quick Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Return",
                        f"{metrics['total_return']:.1%}",
                        delta=f"{metrics['total_return']:.1%}"
                    )

                with col2:
                    st.metric("CAGR", f"{metrics['cagr']:.1%}")

                with col3:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

                with col4:
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")

        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")
            st.exception(e)

    # Show example if no backtest run yet
    if st.session_state.backtest_results is None:
        st.info("👈 Configure your strategy in the sidebar and click 'Run Backtest'")


def show_results_page():
    """Results Dashboard page"""

    st.title("📈 Backtest Results")

    # Check if results exist
    if st.session_state.backtest_results is None:
        st.warning("⚠️ No backtest results available. Please run a backtest first.")
        if st.button("Go to Strategy Builder"):
            pass  # Will switch when user clicks sidebar
        return

    # Load results
    results = st.session_state.backtest_results
    equity_curve = results['equity_curve']
    metrics = results['metrics']
    config = results['strategy_config']

    # Strategy info header
    factor_names = ' + '.join([k.replace('_', ' ').title() for k in config['factors'].keys()])
    st.subheader(f"Strategy: {factor_names}")
    st.caption(f"Universe: {config['universe']} | Period: {config['dates'][0]} to {config['dates'][1]} | Rebalancing: {config['rebalance']}")

    st.markdown("---")

    # Key Metrics
    st.subheader("📊 Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Return",
            f"{metrics['total_return']:.1%}",
            delta=f"{metrics['total_return']:.1%}"
        )
        st.metric("Volatility", f"{metrics['volatility']:.1%}")

    with col2:
        st.metric("CAGR", f"{metrics['cagr']:.1%}")
        st.metric("Win Rate", f"{metrics['win_rate']:.1%}")

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            help="Risk-adjusted return (>1 is good)"
        )
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")

    with col4:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
        st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")

    st.markdown("---")

    # Equity Curve Chart
    st.subheader("📈 Equity Curve")

    fig_equity = go.Figure()

    fig_equity.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00D9FF', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 217, 255, 0.1)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: $%{y:,.0f}<extra></extra>'
    ))

    fig_equity.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )

    st.plotly_chart(fig_equity, use_container_width=True)

    st.markdown("---")

    # Additional analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Returns", "📉 Drawdown", "ℹ️ Details"])

    with tab1:
        st.subheader("Returns Distribution")

        returns = equity_curve['daily_return'].dropna()

        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Daily Returns',
                marker_color='#00D9FF'
            ))
            fig_hist.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Return",
                yaxis_title="Frequency",
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.write("**Returns Statistics**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Min', 'Max'],
                'Value': [
                    f"{returns.mean():.4%}",
                    f"{returns.median():.4%}",
                    f"{returns.std():.4%}",
                    f"{returns.skew():.2f}",
                    f"{returns.min():.4%}",
                    f"{returns.max():.4%}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Drawdown Analysis")

        # Drawdown chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['drawdown'],
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#F44336'),
            fillcolor='rgba(244, 67, 54, 0.3)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2%}<extra></extra>'
        ))
        fig_dd.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            height=350,
            template='plotly_white'
        )
        fig_dd.update_yaxes(tickformat='.0%')

        st.plotly_chart(fig_dd, use_container_width=True)

    with tab3:
        st.subheader("Strategy Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Factor Weights**")
            factors_df = pd.DataFrame(
                list(config['factors'].items()),
                columns=['Factor', 'Weight']
            )
            factors_df['Weight'] = factors_df['Weight'].apply(lambda x: f'{x:.1%}')
            st.dataframe(factors_df, hide_index=True, use_container_width=True)

        with col2:
            st.write("**Configuration**")
            config_df = pd.DataFrame({
                'Setting': ['Universe', 'Start Date', 'End Date', 'Rebalancing', 'Portfolio Type'],
                'Value': [
                    config['universe'],
                    str(config['dates'][0]),
                    str(config['dates'][1]),
                    config['rebalance'],
                    config['portfolio_type']
                ]
            })
            st.dataframe(config_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export Results (JSON)", use_container_width=True):
            export_data = {
                'strategy_config': config,
                'metrics': metrics,
                'equity_curve': equity_curve.to_dict()
            }

            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2, default=str),
                file_name="backtest_results.json",
                mime="application/json"
            )

    with col2:
        if st.button("📥 Export Equity Curve (CSV)", use_container_width=True):
            csv = equity_curve.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="equity_curve.csv",
                mime="text/csv"
            )


def show_factor_explorer():
    """Factor Explorer educational page"""

    st.title("🔍 Factor Explorer")
    st.write("Learn about quantitative factors and their academic foundations")

    st.markdown("---")

    # Factor selection
    factor = st.selectbox(
        "Select a Factor",
        ["Momentum", "Value", "Quality", "Size", "Low Volatility"]
    )

    if factor == "Momentum":
        st.header("📊 Momentum Factor")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Definition")
            st.write("""
            The momentum factor captures the tendency of stocks that have performed well
            in the past to continue performing well in the near future, and vice versa.
            """)

            st.subheader("Calculation")
            st.code("""
# 12-month momentum (skip last month)
momentum = (Price_t / Price_t-252) - 1

# Skip last month to avoid reversal effects
momentum = momentum.shift(21 days)
            """, language="python")

            st.subheader("Academic Research")
            st.write("""
            **Jegadeesh & Titman (1993)** "Returns to Buying Winners and Selling Losers"

            - Found that strategies buying past winners and selling past losers generate
              significant abnormal returns
            - Effect persists for 3-12 months
            - Strongest with 6-12 month formation period
            """)

        with col2:
            st.metric("Typical Return", "8-12% annually")
            st.metric("Sharpe Ratio", "0.6 - 0.8")

            st.info("**Works Best In:**\n- Bull markets\n- Low volatility periods\n- Trending markets")

            st.warning("**Risks:**\n- Momentum crashes\n- Reversals\n- High turnover")

    elif factor == "Value":
        st.header("💰 Value Factor")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Definition")
            st.write("""
            The value factor identifies stocks trading below their intrinsic value,
            typically measured by Price-to-Book, Price-to-Earnings, or other valuation ratios.
            """)

            st.subheader("Calculation")
            st.code("""
# Book-to-Market ratio (inverse of P/B)
value_score = 1 / (Price / Book_Value_Per_Share)

# Higher score = cheaper = more value
            """, language="python")

            st.subheader("Academic Research")
            st.write("""
            **Fama & French (1993)** "Common Risk Factors in Returns"

            - Value stocks (high book-to-market) outperform growth stocks
            - Effect is stronger for small-cap stocks
            - Average value premium: ~5% per year
            """)

        with col2:
            st.metric("Value Premium", "4-6% annually")
            st.metric("Sharpe Ratio", "0.3 - 0.5")

            st.info("**Works Best In:**\n- Recovery periods\n- Mean reversion\n- Economic expansions")

    # Add other factors similarly...

    st.markdown("---")
    st.subheader("📚 Recommended Reading")
    st.write("""
    **Books:**
    - "Quantitative Momentum" by Wesley Gray
    - "Your Complete Guide to Factor-Based Investing" by Andrew Berkin

    **Papers:**
    - Fama & French (1993) - Three-Factor Model
    - Jegadeesh & Titman (1993) - Momentum
    - Novy-Marx (2013) - Quality Factor
    """)


if __name__ == "__main__":
    main()
