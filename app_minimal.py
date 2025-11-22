"""
🤖 Agentic Backtesting Framework - Minimalist Edition

Ultra-clean, minimalist interface with navy and white color scheme.
Focus on data, clarity, and simplicity.

Author: Agentic Backtesting Framework
Version: 3.0 Minimal
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
from core.indicators import TechnicalIndicators
from core.feature_engineering import FeatureEngineeringPipeline
from core.agent import AgenticStrategyExplorer
from core.analytics import PerformanceAnalytics

# Page configuration
st.set_page_config(
    page_title="Agentic Quant Trading",
    page_icon="▪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist CSS - Navy and White Only
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background-color: #ffffff;
        color: #1a1f36;
    }

    /* Clean header */
    .minimal-header {
        font-size: 2.5rem;
        font-weight: 300;
        color: #1a1f36;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1a1f36;
    }

    .minimal-subtitle {
        font-size: 1rem;
        font-weight: 300;
        color: #6b7280;
        margin-bottom: 3rem;
    }

    /* Section headers */
    .section-title {
        font-size: 1.5rem;
        font-weight: 400;
        color: #1a1f36;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }

    /* Clean cards */
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #1a1f36;
        padding: 2rem 1.5rem;
        margin: 1rem 0;
        transition: all 0.2s ease;
    }

    .metric-box:hover {
        box-shadow: 0 0 0 1px #1a1f36;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 300;
        color: #1a1f36;
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.875rem;
        font-weight: 400;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Info boxes */
    .info-box {
        background-color: #f9fafb;
        border-left: 3px solid #1a1f36;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }

    .info-box-title {
        font-size: 1rem;
        font-weight: 500;
        color: #1a1f36;
        margin-bottom: 0.5rem;
    }

    .info-box-text {
        font-size: 0.9rem;
        color: #6b7280;
        line-height: 1.6;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1a1f36;
        color: #ffffff;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 400;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        border-radius: 0;
    }

    .stButton>button:hover {
        background-color: #2d3748;
        box-shadow: none;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1f36;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        background-color: transparent;
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #ffffff;
    }

    [data-testid="stSidebar"] input[type="radio"]:checked + label {
        background-color: rgba(255, 255, 255, 0.1);
        border-left: 3px solid #ffffff;
    }

    /* Clean divider */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }

    /* Tables */
    .dataframe {
        border: 1px solid #e5e7eb !important;
        font-size: 0.9rem;
    }

    .dataframe th {
        background-color: #1a1f36 !important;
        color: #ffffff !important;
        font-weight: 400;
        padding: 0.75rem !important;
    }

    .dataframe td {
        padding: 0.75rem !important;
        border-bottom: 1px solid #e5e7eb !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 300;
        color: #1a1f36;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 400;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Select boxes */
    .stSelectbox label {
        font-size: 0.875rem;
        font-weight: 400;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sliders */
    .stSlider label {
        font-size: 0.875rem;
        font-weight: 400;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Remove default styling */
    .element-container {
        margin: 0;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #1a1f36;
    }

    /* Clean status messages */
    .stSuccess, .stInfo, .stWarning {
        background-color: #f9fafb;
        border-left: 3px solid #1a1f36;
        color: #1a1f36;
        padding: 1rem;
        border-radius: 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'explorer' not in st.session_state:
    st.session_state.explorer = None
if 'strategies' not in st.session_state:
    st.session_state.strategies = []


def load_korean_stock_data():
    """Load Korean stock data."""
    try:
        progress = st.progress(0)
        status = st.empty()

        status.text("Loading data...")
        progress.progress(25)

        data_path = Path("data/processed/stock_prices_clean.csv")

        if data_path.exists():
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            progress.progress(50)

            df_long = df.stack().reset_index()
            df_long.columns = ['date', 'ticker', 'close']
            df_long = df_long.set_index(['ticker', 'date'])
            progress.progress(75)

            df_long['open'] = df_long['close']
            df_long['high'] = df_long['close'] * 1.02
            df_long['low'] = df_long['close'] * 0.98
            df_long['volume'] = 1000000
            progress.progress(100)

            status.success("Data loaded")
            return df_long
        else:
            st.error("Data file not found")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def display_home():
    """Minimalist home page."""

    st.markdown('<div class="minimal-header">Agentic Quant Trading</div>', unsafe_allow_html=True)
    st.markdown('<div class="minimal-subtitle">Autonomous strategy discovery for Korean stock market</div>', unsafe_allow_html=True)

    # Stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">20+</div>
            <div class="metric-label">Indicators</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">4</div>
            <div class="metric-label">Algorithms</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">2,620</div>
            <div class="metric-label">Stocks</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">∞</div>
            <div class="metric-label">Strategies</div>
        </div>
        """, unsafe_allow_html=True)

    # Features
    st.markdown('<div class="section-title">Features</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">Autonomous Exploration</div>
        <div class="info-box-text">
            AI agent explores thousands of parameter combinations to discover optimal strategies automatically.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">Korean Market Optimized</div>
        <div class="info-box-text">
            Mean reversion detection, momentum reversals, and market-specific analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">Advanced Analytics</div>
        <div class="info-box-text">
            Geometric returns, Sharpe/Sortino/Calmar ratios, risk metrics, and performance tracking.
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_data_explorer():
    """Minimalist data explorer."""

    st.markdown('<div class="minimal-header">Data Explorer</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.info("Load Korean stock data from sidebar to continue")
        return

    data = st.session_state.data
    tickers = sorted(data.index.get_level_values(0).unique())

    selected_ticker = st.selectbox("Select Stock", tickers, label_visibility="collapsed")

    if selected_ticker:
        ticker_data = data.loc[selected_ticker]

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        current_price = ticker_data['close'].iloc[-1]
        price_change = ticker_data['close'].pct_change().iloc[-1]
        high_52w = ticker_data['close'].rolling(252).max().iloc[-1]
        low_52w = ticker_data['close'].rolling(252).min().iloc[-1]

        with col1:
            st.metric("Price", f"₩{current_price:,.0f}", f"{price_change:.2%}")
        with col2:
            st.metric("52W High", f"₩{high_52w:,.0f}")
        with col3:
            st.metric("52W Low", f"₩{low_52w:,.0f}")
        with col4:
            vol = ticker_data['close'].pct_change().std() * np.sqrt(252)
            st.metric("Volatility", f"{vol:.2%}")

        # Chart
        st.markdown('<div class="section-title">Technical Analysis</div>', unsafe_allow_html=True)

        indicators = TechnicalIndicators(ticker_data)

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price', 'RSI', 'MACD', 'Volume'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )

        # Price
        fig.add_trace(
            go.Scatter(
                x=ticker_data.index,
                y=ticker_data['close'],
                name='Price',
                line=dict(color='#1a1f36', width=1.5)
            ),
            row=1, col=1
        )

        # SMA
        fig.add_trace(
            go.Scatter(
                x=ticker_data.index,
                y=indicators.sma(20),
                name='SMA 20',
                line=dict(color='#1a1f36', width=1, dash='dash')
            ),
            row=1, col=1
        )

        # RSI
        rsi = indicators.rsi()
        fig.add_trace(
            go.Scatter(
                x=ticker_data.index,
                y=rsi,
                name='RSI',
                line=dict(color='#1a1f36', width=1)
            ),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#9ca3af", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#9ca3af", row=2, col=1)

        # MACD
        macd_df = indicators.macd()
        fig.add_trace(
            go.Scatter(
                x=ticker_data.index,
                y=macd_df['macd'],
                name='MACD',
                line=dict(color='#1a1f36', width=1)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=ticker_data.index,
                y=macd_df['signal'],
                name='Signal',
                line=dict(color='#6b7280', width=1, dash='dash')
            ),
            row=3, col=1
        )

        # Volume
        fig.add_trace(
            go.Bar(
                x=ticker_data.index,
                y=ticker_data['volume'],
                name='Volume',
                marker_color='#e5e7eb'
            ),
            row=4, col=1
        )

        fig.update_layout(
            height=900,
            showlegend=False,
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(family='Inter', color='#1a1f36', size=11),
            margin=dict(l=10, r=10, t=30, b=10)
        )

        fig.update_xaxes(showgrid=False, showline=True, linecolor='#e5e7eb')
        fig.update_yaxes(showgrid=True, gridcolor='#f3f4f6', showline=True, linecolor='#e5e7eb')

        st.plotly_chart(fig, use_container_width=True)


def display_optimization():
    """Minimalist optimization page."""

    st.markdown('<div class="minimal-header">Strategy Optimization</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.info("Load Korean stock data from sidebar to continue")
        return

    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox("Method", ["genetic", "random", "grid"])
    with col2:
        n_strategies = st.slider("Strategies", 3, 20, 5)
    with col3:
        budget = st.slider("Budget", 50, 500, 200, 50)

    objective = st.selectbox("Objective", ["sharpe", "sortino", "calmar", "returns"])

    if st.button("Run Optimization", type="primary", use_container_width=True):
        with st.spinner("Optimizing..."):
            try:
                explorer = AgenticStrategyExplorer(
                    data=st.session_state.data,
                    objective=objective,
                    korean_market=True
                )

                strategies = explorer.explore(
                    method=method,
                    n_strategies=n_strategies,
                    exploration_budget=budget,
                    verbose=False
                )

                st.session_state.explorer = explorer
                st.session_state.strategies = strategies

                st.success(f"Discovered {len(strategies)} strategies")

                comparison = explorer.get_strategy_comparison()
                st.dataframe(comparison, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")


def main():
    """Main application."""

    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")

        page = st.radio(
            "",
            ["Home", "Data Explorer", "Optimization"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        if st.button("Load Data", use_container_width=True):
            data = load_korean_stock_data()
            if data is not None:
                st.session_state.data = data
                st.session_state.data_loaded = True

        if st.session_state.data_loaded:
            n_stocks = len(st.session_state.data.index.get_level_values(0).unique())
            st.caption(f"{n_stocks} stocks loaded")

    # Routing
    if page == "Home":
        display_home()
    elif page == "Data Explorer":
        display_data_explorer()
    elif page == "Optimization":
        display_optimization()


if __name__ == "__main__":
    main()
