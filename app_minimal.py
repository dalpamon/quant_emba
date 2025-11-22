"""
🤖 Agentic Backtesting Framework - Minimalist Edition

Ultra-clean, minimalist interface with navy and white color scheme.
Focus on data, clarity, and simplicity.

Features:
- Stock Screener with live data
- Real-time data from Yahoo Finance
- Technical indicators for all stocks

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
import yfinance as yf
from datetime import datetime, timedelta

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

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

    /* Stock card */
    .stock-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }

    .stock-card:hover {
        border-color: #1a1f36;
    }

    .stock-name {
        font-size: 1rem;
        font-weight: 500;
        color: #1a1f36;
        margin-bottom: 0.25rem;
    }

    .stock-ticker {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
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
if 'live_data' not in st.session_state:
    st.session_state.live_data = None
if 'explorer' not in st.session_state:
    st.session_state.explorer = None
if 'strategies' not in st.session_state:
    st.session_state.strategies = []

# Korean stock tickers (Top 30)
KOREAN_TICKERS = [
    '005930.KS',  # Samsung Electronics
    '000660.KS',  # SK Hynix
    '373220.KS',  # LG Energy Solution
    '207940.KS',  # Samsung Biologics
    '005490.KS',  # POSCO Holdings
    '051910.KS',  # LG Chem
    '006400.KS',  # Samsung SDI
    '035420.KS',  # NAVER
    '000270.KS',  # Kia
    '005380.KS',  # Hyundai Motor
    '068270.KS',  # Celltrion
    '035720.KS',  # Kakao
    '105560.KS',  # KB Financial
    '055550.KS',  # Shinhan Financial
    '012330.KS',  # Hyundai Mobis
    '028260.KS',  # Samsung C&T
    '066570.KS',  # LG Electronics
    '323410.KS',  # Kakao Bank
    '003550.KS',  # LG
    '017670.KS',  # SK Telecom
    '096770.KS',  # SK Innovation
    '034730.KS',  # SK
    '009150.KS',  # Samsung Electro-Mechanics
    '018260.KS',  # Samsung SDS
    '086790.KS',  # Hana Financial Group
    '032830.KS',  # Samsung Life Insurance
    '010130.KS',  # Korea Zinc
    '003670.KS',  # POSCO INTERNATIONAL
    '047810.KS',  # Korea Aerospace Industries
    '036570.KS',  # NCSOFT
]


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_live_data(tickers, period='1y'):
    """Fetch live data from Yahoo Finance."""
    try:
        data_frames = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)

                if not hist.empty:
                    hist['ticker'] = ticker
                    hist = hist.reset_index()
                    hist.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'ticker']
                    hist = hist[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
                    data_frames.append(hist)
            except:
                continue

        if data_frames:
            combined = pd.concat(data_frames, ignore_index=True)
            combined = combined.set_index(['ticker', 'date'])
            return combined
        return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def load_korean_stock_data():
    """Load Korean stock data from file or fetch live."""
    try:
        # Try loading from file first
        data_path = Path("data/processed/stock_prices_clean.csv")

        if data_path.exists():
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            df_long = df.stack().reset_index()
            df_long.columns = ['date', 'ticker', 'close']
            df_long = df_long.set_index(['ticker', 'date'])
            df_long['open'] = df_long['close']
            df_long['high'] = df_long['close'] * 1.02
            df_long['low'] = df_long['close'] * 0.98
            df_long['volume'] = 1000000
            return df_long
        else:
            # Fall back to live data
            st.info("Local data not found. Fetching live data from Yahoo Finance...")
            return fetch_live_data(KOREAN_TICKERS)
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def calculate_stock_metrics(ticker_data):
    """Calculate key metrics for a stock."""
    try:
        indicators = TechnicalIndicators(ticker_data)

        current_price = ticker_data['close'].iloc[-1]
        price_change = ticker_data['close'].pct_change().iloc[-1]

        rsi = indicators.rsi().iloc[-1]
        macd_df = indicators.macd()
        macd = macd_df['macd'].iloc[-1]

        vol = ticker_data['close'].pct_change().std() * np.sqrt(252)

        return {
            'price': current_price,
            'change': price_change,
            'rsi': rsi,
            'macd': macd,
            'volatility': vol
        }
    except:
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
            <div class="metric-value">30</div>
            <div class="metric-label">Korean Stocks</div>
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
        <div class="info-box-title">Live Market Data</div>
        <div class="info-box-text">
            Real-time data from Yahoo Finance. Always up-to-date with latest prices and indicators.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">Stock Screener</div>
        <div class="info-box-text">
            View all stocks with RSI, MACD, and key metrics in one dashboard. Filter and sort by indicators.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">Autonomous Optimization</div>
        <div class="info-box-text">
            AI agent explores thousands of parameter combinations to discover optimal strategies automatically.
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_stock_screener():
    """Stock screener showing all stocks with indicators."""

    st.markdown('<div class="minimal-header">Stock Screener</div>', unsafe_allow_html=True)
    st.markdown('<div class="minimal-subtitle">Live market data with technical indicators</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.info("Load market data from sidebar to continue")
        return

    data = st.session_state.live_data
    tickers = sorted(data.index.get_level_values(0).unique())

    # Calculate metrics for all stocks
    st.markdown('<div class="section-title">Market Overview</div>', unsafe_allow_html=True)

    metrics_data = []

    progress_bar = st.progress(0)
    for i, ticker in enumerate(tickers):
        try:
            ticker_data = data.loc[ticker]
            metrics = calculate_stock_metrics(ticker_data)

            if metrics:
                metrics_data.append({
                    'Ticker': ticker.replace('.KS', ''),
                    'Price': f"₩{metrics['price']:,.0f}",
                    'Change %': f"{metrics['change']:.2%}",
                    'RSI': f"{metrics['rsi']:.1f}",
                    'MACD': f"{metrics['macd']:.4f}",
                    'Volatility': f"{metrics['volatility']:.2%}",
                    'RSI_val': metrics['rsi'],  # For filtering
                })
        except:
            continue

        progress_bar.progress((i + 1) / len(tickers))

    progress_bar.empty()

    if metrics_data:
        df = pd.DataFrame(metrics_data)

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            rsi_filter = st.selectbox(
                "RSI Filter",
                ["All", "Oversold (<30)", "Neutral (30-70)", "Overbought (>70)"]
            )

        with col2:
            sort_by = st.selectbox(
                "Sort By",
                ["Ticker", "RSI", "Change %", "Volatility"]
            )

        with col3:
            ascending = st.checkbox("Ascending", value=True)

        # Apply filters
        if rsi_filter == "Oversold (<30)":
            df = df[df['RSI_val'] < 30]
        elif rsi_filter == "Neutral (30-70)":
            df = df[(df['RSI_val'] >= 30) & (df['RSI_val'] <= 70)]
        elif rsi_filter == "Overbought (>70)":
            df = df[df['RSI_val'] > 70]

        # Sort
        if sort_by == "RSI":
            df = df.sort_values('RSI_val', ascending=ascending)
        elif sort_by != "Ticker":
            df = df.sort_values(sort_by, ascending=ascending)

        # Display table
        display_df = df.drop(columns=['RSI_val'])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Summary stats
        st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            oversold = len(df[df['RSI_val'] < 30])
            st.metric("Oversold Stocks", oversold)

        with col2:
            overbought = len(df[df['RSI_val'] > 70])
            st.metric("Overbought Stocks", overbought)

        with col3:
            avg_rsi = df['RSI_val'].mean()
            st.metric("Average RSI", f"{avg_rsi:.1f}")

        with col4:
            total_stocks = len(df)
            st.metric("Total Stocks", total_stocks)


def display_data_explorer():
    """Minimalist data explorer."""

    st.markdown('<div class="minimal-header">Data Explorer</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.info("Load market data from sidebar to continue")
        return

    data = st.session_state.live_data
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
        st.info("Load market data from sidebar to continue")
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
                    data=st.session_state.live_data,
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
            ["Home", "Stock Screener", "Data Explorer", "Optimization"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        if st.button("Load Live Data", use_container_width=True):
            with st.spinner("Fetching live data..."):
                data = fetch_live_data(KOREAN_TICKERS, period='1y')
                if data is not None:
                    st.session_state.live_data = data
                    st.session_state.data_loaded = True
                    st.success("Live data loaded")
                else:
                    # Fallback to local data
                    data = load_korean_stock_data()
                    if data is not None:
                        st.session_state.live_data = data
                        st.session_state.data_loaded = True
                        st.success("Local data loaded")

        if st.session_state.data_loaded:
            n_stocks = len(st.session_state.live_data.index.get_level_values(0).unique())
            st.caption(f"{n_stocks} stocks loaded")

            last_update = datetime.now().strftime("%H:%M")
            st.caption(f"Updated: {last_update}")

    # Routing
    if page == "Home":
        display_home()
    elif page == "Stock Screener":
        display_stock_screener()
    elif page == "Data Explorer":
        display_data_explorer()
    elif page == "Optimization":
        display_optimization()


if __name__ == "__main__":
    main()
