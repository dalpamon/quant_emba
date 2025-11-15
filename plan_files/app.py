"""
Factor Lab - Streamlit Web Application
Interactive backtesting platform for quantitative strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime
import json

from factors import FactorEngine
from portfolio import Portfolio
from backtest import Backtester
from analytics import PerformanceAnalytics
from data_loader import DataLoader


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
if 'equity_curve' not in st.session_state:
    st.session_state.equity_curve = None
if 'positions_history' not in st.session_state:
    st.session_state.positions_history = None


def load_data_from_db():
    """Load data from database"""
    conn = sqlite3.connect('quant1_data.db')
    
    # Load universe
    universe = pd.read_sql("SELECT * FROM quant1_universe WHERE active = 1", conn)
    
    # Load prices
    prices = pd.read_sql("SELECT * FROM quant1_prices", conn)
    
    # Load factors
    factors = pd.read_sql("SELECT * FROM quant1_factors", conn)
    
    conn.close()
    
    return universe, prices, factors


def create_equity_curve_chart(equity_curve, benchmark=None):
    """Create interactive equity curve chart"""
    
    fig = go.Figure()
    
    # Strategy line
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Strategy',
        line=dict(color='#00D9FF', width=2),
        hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>'
    ))
    
    # Benchmark line (if provided)
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_drawdown_chart(equity_curve):
    """Create drawdown chart"""
    
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#FF6B6B', width=0),
        fillcolor='rgba(255, 107, 107, 0.3)',
        hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=350
    )
    
    return fig


def create_monthly_returns_heatmap(equity_curve):
    """Create monthly returns heatmap"""
    
    returns = equity_curve.pct_change()
    
    # Group by year and month
    monthly_returns = returns.groupby([returns.index.year, returns.index.month]).sum() * 100
    
    # Reshape to matrix
    monthly_data = monthly_returns.unstack(fill_value=0)
    
    if monthly_data.empty:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=monthly_data.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=monthly_data.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(monthly_data.values, 2),
        texttemplate='%{text}%',
        textfont={"size": 10},
        hovertemplate='<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Monthly Returns Heatmap (%)",
        height=400
    )
    
    return fig


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧪 Factor Lab</h1>', unsafe_allow_html=True)
    st.markdown("### Multi-Factor Backtesting Engine")
    st.markdown("---")
    
    # Check if database exists
    try:
        universe, prices, factors = load_data_from_db()
    except Exception as e:
        st.error("""
        ⚠️ Database not found or empty!
        
        Please run setup first:
        ```
        python setup_database.py
        ```
        """)
        st.stop()
    
    # Sidebar - Strategy Configuration
    st.sidebar.header("🎯 Strategy Builder")
    
    # Universe selection
    available_tickers = universe['ticker'].unique().tolist()
    st.sidebar.markdown(f"**Available Stocks:** {len(available_tickers)}")
    
    # Date range
    st.sidebar.markdown("### 📅 Date Range")
    
    min_date = pd.to_datetime(prices['date'].min())
    max_date = pd.to_datetime(prices['date'].max())
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    # Factor weights
    st.sidebar.markdown("### ⚖️ Factor Weights")
    
    momentum_weight = st.sidebar.slider("Momentum (12M)", 0.0, 1.0, 0.4, 0.05)
    value_weight = st.sidebar.slider("Value (B/M)", 0.0, 1.0, 0.3, 0.05)
    quality_weight = st.sidebar.slider("Quality (ROE)", 0.0, 1.0, 0.3, 0.05)
    
    # Normalize weights
    total_weight = momentum_weight + value_weight + quality_weight
    if total_weight > 0:
        factor_weights = {
            'momentum_12m': momentum_weight / total_weight,
            'value_pb': value_weight / total_weight,
            'quality_roe': quality_weight / total_weight
        }
    else:
        st.sidebar.error("Total weight must be > 0")
        st.stop()
    
    st.sidebar.markdown(f"**Normalized Weights:**")
    for factor, weight in factor_weights.items():
        st.sidebar.text(f"{factor}: {weight:.2%}")
    
    # Portfolio parameters
    st.sidebar.markdown("### 📊 Portfolio Parameters")
    
    rebalance_freq = st.sidebar.selectbox(
        "Rebalance Frequency",
        ["M", "Q", "W"],
        index=0,
        format_func=lambda x: {"M": "Monthly", "Q": "Quarterly", "W": "Weekly"}[x]
    )
    
    long_pct = st.sidebar.slider("Long % (Top)", 0.1, 0.5, 0.2, 0.05)
    short_pct = st.sidebar.slider("Short % (Bottom)", 0.1, 0.5, 0.2, 0.05)
    
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    transaction_cost = st.sidebar.slider(
        "Transaction Cost (bps)",
        0, 50, 10, 5
    )
    
    # Run backtest button
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
        
        with st.spinner("Running backtest... This may take a minute..."):
            
            # Prepare data
            prices_pivot = prices.pivot(index='date', columns='ticker', values='adjusted_close')
            prices_pivot.index = pd.to_datetime(prices_pivot.index)
            
            factors_pivot = factors.pivot(index='date', columns='ticker', values='momentum_12m')
            factors_pivot.index = pd.to_datetime(factors_pivot.index)
            
            # Create composite factor score
            factor_dfs = {}
            for factor_name in factor_weights.keys():
                factor_dfs[factor_name] = factors.pivot(
                    index='date', columns='ticker', values=factor_name
                )
                factor_dfs[factor_name].index = pd.to_datetime(factor_dfs[factor_name].index)
            
            # Calculate composite scores for each date
            composite_scores = pd.DataFrame(index=factors_pivot.index, columns=factors_pivot.columns)
            
            for date in composite_scores.index:
                date_factors = {
                    name: df.loc[date] if date in df.index else pd.Series()
                    for name, df in factor_dfs.items()
                }
                
                if all(not s.empty for s in date_factors.values()):
                    composite_scores.loc[date] = FactorEngine.composite_score(
                        date_factors, factor_weights
                    )
            
            # Run backtest
            backtester = Backtester(prices_pivot, composite_scores, initial_capital)
            
            equity_curve, positions_history = backtester.run(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                rebalance_freq=rebalance_freq,
                top_pct=long_pct,
                bottom_pct=short_pct,
                transaction_cost_bps=transaction_cost
            )
            
            # Calculate performance metrics
            report = PerformanceAnalytics.generate_report(equity_curve)
            
            # Store in session state
            st.session_state.equity_curve = equity_curve
            st.session_state.backtest_results = report
            st.session_state.positions_history = positions_history
        
        st.success("✅ Backtest complete!")
        st.rerun()
    
    # Main content area
    if st.session_state.equity_curve is not None:
        
        equity_curve = st.session_state.equity_curve
        report = st.session_state.backtest_results
        
        # Performance metrics
        st.markdown("## 📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{report['Total Return']*100:.2f}%",
                delta=None
            )
            st.metric(
                "CAGR",
                f"{report['CAGR']*100:.2f}%"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{report['Sharpe Ratio']:.2f}"
            )
            st.metric(
                "Sortino Ratio",
                f"{report['Sortino Ratio']:.2f}"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{report['Max Drawdown']*100:.2f}%",
                delta=None,
                delta_color="inverse"
            )
            st.metric(
                "Volatility",
                f"{report['Volatility']*100:.2f}%"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{report['Win Rate']*100:.1f}%"
            )
            st.metric(
                "Calmar Ratio",
                f"{report['Calmar Ratio']:.2f}"
            )
        
        st.markdown("---")
        
        # Charts
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Equity Curve",
            "📉 Drawdown",
            "🔥 Monthly Returns",
            "📋 Positions"
        ])
        
        with tab1:
            st.plotly_chart(
                create_equity_curve_chart(equity_curve),
                use_container_width=True
            )
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Day", f"{report['Best Day']*100:.2f}%")
            with col2:
                st.metric("Worst Day", f"{report['Worst Day']*100:.2f}%")
            with col3:
                st.metric("Profit Factor", f"{report['Profit Factor']:.2f}")
        
        with tab2:
            st.plotly_chart(
                create_drawdown_chart(equity_curve),
                use_container_width=True
            )
            
            # Drawdown analysis
            st.markdown("### Drawdown Periods")
            dd_analysis = PerformanceAnalytics.drawdown_analysis(equity_curve)
            if not dd_analysis.empty:
                st.dataframe(dd_analysis, use_container_width=True)
            else:
                st.info("No significant drawdown periods")
        
        with tab3:
            heatmap = create_monthly_returns_heatmap(equity_curve)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)
            else:
                st.info("Insufficient data for monthly heatmap")
        
        with tab4:
            st.markdown("### Latest Positions")
            
            if st.session_state.positions_history:
                latest_date = max(st.session_state.positions_history.keys())
                latest_positions = st.session_state.positions_history[latest_date]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🟢 Long Positions**")
                    long_df = pd.DataFrame([
                        {
                            'Ticker': ticker,
                            'Weight': f"{data['weight']*100:.2f}%",
                            'Score': f"{data['score']:.3f}",
                            'Price': f"${data['price']:.2f}" if data['price'] else "N/A"
                        }
                        for ticker, data in latest_positions['long'].items()
                    ])
                    if not long_df.empty:
                        st.dataframe(long_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No long positions")
                
                with col2:
                    st.markdown("**🔴 Short Positions**")
                    short_df = pd.DataFrame([
                        {
                            'Ticker': ticker,
                            'Weight': f"{data['weight']*100:.2f}%",
                            'Score': f"{data['score']:.3f}",
                            'Price': f"${data['price']:.2f}" if data['price'] else "N/A"
                        }
                        for ticker, data in latest_positions['short'].items()
                    ])
                    if not short_df.empty:
                        st.dataframe(short_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No short positions")
                
                st.markdown(f"*Rebalanced on: {latest_date.strftime('%Y-%m-%d')}*")
    
    else:
        # Welcome screen
        st.info("""
        ### 👋 Welcome to Factor Lab!
        
        Configure your strategy in the sidebar and click **Run Backtest** to begin.
        
        **Quick Start:**
        1. Adjust factor weights (Momentum, Value, Quality)
        2. Set portfolio parameters (long/short percentages)
        3. Choose rebalancing frequency
        4. Click "Run Backtest"
        
        **Features:**
        - Multi-factor portfolio construction
        - Long-short strategies
        - Realistic transaction costs
        - Comprehensive performance analytics
        """)
        
        # Show universe info
        st.markdown("### 📊 Current Universe")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks", len(universe))
        with col2:
            st.metric("Date Range", f"{min_date.year} - {max_date.year}")
        with col3:
            sectors = universe['sector'].value_counts()
            st.metric("Sectors", len(sectors))
        
        # Top stocks by market cap
        st.markdown("### 🏆 Top 10 Stocks by Market Cap")
        top10 = universe.nlargest(10, 'market_cap')[['ticker', 'name', 'sector', 'market_cap']]
        top10['market_cap'] = top10['market_cap'].apply(lambda x: f"${x/1e9:.2f}B")
        st.dataframe(top10, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
