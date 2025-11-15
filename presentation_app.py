"""
Interactive Web Presentation: Korean Stock Market Momentum Analysis
====================================================================
Streamlit app for presenting momentum strategy results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Korean Momentum Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main content styles */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .highlight-green {
        color: #2ecc71;
        font-weight: bold;
    }
    .highlight-red {
        color: #e74c3c;
        font-weight: bold;
    }
    .stAlert {
        margin-top: 1rem;
    }

    /* Sidebar styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    /* Hide radio buttons */
    [data-testid="stSidebar"] .stRadio {
        display: none;
    }

    /* Navigation button styles */
    .nav-button {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: white;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: left;
        width: 100%;
        display: block;
    }

    .nav-button:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
        transform: translateX(5px);
    }

    .nav-button.active {
        background: rgba(255, 255, 255, 0.25);
        border-color: #4CAF50;
        border-left: 4px solid #4CAF50;
        font-weight: 600;
    }

    .nav-section {
        color: rgba(255, 255, 255, 0.6);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 10px 0;
        padding-left: 16px;
    }

    .sidebar-header {
        background: rgba(0, 0, 0, 0.2);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }

    .sidebar-header h1 {
        color: white;
        font-size: 24px;
        margin: 0;
        font-weight: 700;
    }

    .sidebar-header p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 14px;
        margin: 8px 0 0 0;
    }

    .info-box {
        background: rgba(76, 175, 80, 0.2);
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
        color: white;
    }

    .info-box p {
        margin: 5px 0;
        font-size: 13px;
        color: rgba(255, 255, 255, 0.9);
    }

    .info-box strong {
        color: white;
        font-weight: 600;
    }

    /* Streamlit button override for sidebar */
    [data-testid="stSidebar"] button {
        background: transparent !important;
        border: none !important;
        color: white !important;
        padding: 0 !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load all analysis results"""
    try:
        backtest_results = pd.read_csv('output/results/momentum_backtest_results.csv',
                                       index_col=0, parse_dates=True)
        summary_stats = pd.read_csv('output/results/momentum_summary.csv')
        stock_stats = pd.read_csv('output/results/stock_summary_stats.csv', index_col=0)
        stock_prices = pd.read_csv('data/processed/stock_prices_clean.csv',
                                   index_col=0, parse_dates=True)

        return backtest_results, summary_stats, stock_stats, stock_prices
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Load data
backtest_results, summary_stats, stock_stats, stock_prices = load_data()

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sector classification function
@st.cache_data
def classify_sector(stock_name):
    """Classify stocks into sectors based on company name"""
    stock_lower = stock_name.lower()

    # Semiconductor
    if any(keyword in stock_lower for keyword in ['반도체', '하이닉스', 'skhynix', '삼성전자', 'sk하이닉스']):
        return 'Semiconductor'

    # AI/Tech
    if any(keyword in stock_lower for keyword in ['ai', 'naver', '네이버', '카카오', 'kakao', '엔씨소프트', 'nc', '넷마블', '크래프톤']):
        return 'AI/Tech'

    # Battery/Energy
    if any(keyword in stock_lower for keyword in ['에너지', 'energy', '배터리', 'battery', 'lg에너지', 'sk온', '삼성sdi']):
        return 'Battery/Energy'

    # Bio/Pharma
    if any(keyword in stock_lower for keyword in ['바이오', 'bio', '제약', 'pharma', '셀트리온', '삼성바이오']):
        return 'Bio/Pharma'

    # Auto/Manufacturing
    if any(keyword in stock_lower for keyword in ['현대차', '기아', 'hyundai', 'kia', '모비스', 'mobis']):
        return 'Auto'

    # Shipbuilding
    if any(keyword in stock_lower for keyword in ['조선', 'shipbuilding', 'ship', '중공업', 'heavy']):
        return 'Shipbuilding'

    # Aerospace/Defense
    if any(keyword in stock_lower for keyword in ['항공', 'aerospace', '에어로스페이스', '방산', 'defense']):
        return 'Aerospace/Defense'

    # Finance
    if any(keyword in stock_lower for keyword in ['금융', '은행', 'bank', '증권', '보험', 'insurance', '카드', 'kb', '신한', '하나']):
        return 'Finance'

    # Chemical
    if any(keyword in stock_lower for keyword in ['화학', 'chemical', 'lg화학', '롯데케미칼']):
        return 'Chemical'

    # Steel/Metal
    if any(keyword in stock_lower for keyword in ['철강', 'steel', 'posco', '아연', 'zinc']):
        return 'Steel/Metal'

    # Retail/Commerce
    if any(keyword in stock_lower for keyword in ['유통', 'retail', '마트', '이마트', 'emart', '롯데쇼핑']):
        return 'Retail'

    # Construction
    if any(keyword in stock_lower for keyword in ['건설', 'construction', '엔지니어링', 'engineering']):
        return 'Construction'

    # Entertainment
    if any(keyword in stock_lower for keyword in ['엔터', 'entertainment', 'sm', 'jyp', 'yg', 'hybe']):
        return 'Entertainment'

    # Food/Beverage
    if any(keyword in stock_lower for keyword in ['식품', 'food', '음료', 'beverage', '푸드']):
        return 'Food/Beverage'

    # Telecom
    if any(keyword in stock_lower for keyword in ['통신', 'telecom', 'kt', 'skt', '텔레콤']):
        return 'Telecom'

    # Utility
    if any(keyword in stock_lower for keyword in ['전력', 'power', 'electric', '한국전력', 'kepco', '가스', 'gas']):
        return 'Utility'

    return 'Other'

# Alpha and Beta calculation function
@st.cache_data
def calculate_alpha_beta(stock_prices, risk_free_rate=0.035):
    """
    Calculate alpha and beta for all stocks using regression

    Regression: stock_returns = alpha + beta × market_returns + error
    - Alpha (α) = intercept = excess return (skill/luck)
    - Beta (β) = slope = market sensitivity (systematic risk)

    Parameters:
    - stock_prices: DataFrame of stock prices
    - risk_free_rate: Annual risk-free rate (default 3.5% for Korean bonds)
    """
    from scipy import stats

    # Calculate returns
    returns = stock_prices.pct_change().dropna()

    # Create market proxy (equal-weighted average of all stocks)
    market_returns = returns.mean(axis=1)

    # Daily risk-free rate
    rf_daily = risk_free_rate / 252

    # Calculate alpha and beta for each stock
    alphas = {}
    betas = {}
    r_squared = {}

    for stock in returns.columns:
        stock_returns = returns[stock].dropna()

        # Align data
        common_idx = stock_returns.index.intersection(market_returns.index)
        if len(common_idx) > 30:  # Need enough data points
            stock_ret = stock_returns[common_idx]
            mkt_ret = market_returns[common_idx]

            # Excess returns (subtract risk-free rate)
            stock_excess = stock_ret - rf_daily
            market_excess = mkt_ret - rf_daily

            # Linear regression: stock_excess = alpha + beta × market_excess
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                market_excess,
                stock_excess
            )

            # Store results
            betas[stock] = slope  # Beta = slope
            alphas[stock] = intercept * 252 * 100  # Annualized alpha in %
            r_squared[stock] = r_value ** 2
        else:
            betas[stock] = np.nan
            alphas[stock] = np.nan
            r_squared[stock] = np.nan

    return pd.Series(alphas), pd.Series(betas), pd.Series(r_squared), market_returns

# Calculate alpha, beta, and R² if data is available
if stock_prices is not None:
    stock_alphas, stock_betas, stock_r_squared, market_returns = calculate_alpha_beta(stock_prices)
else:
    stock_alphas, stock_betas, stock_r_squared, market_returns = None, None, None, None

# Sidebar navigation with custom styling
with st.sidebar:
    # Header
    st.markdown("""
    <div class='sidebar-header'>
        <h1>Korean Momentum Analysis</h1>
        <p>Interactive Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation sections
    st.markdown("<div class='nav-section'>Main Sections</div>", unsafe_allow_html=True)

    pages = [
        ("Home", "home"),
        ("Key Findings", "findings"),
        ("Detailed Results", "results"),
        ("Data Explorer", "explorer"),
    ]

    for page_name, page_key in pages:
        active_class = "active" if st.session_state.page == page_name else ""
        if st.button(page_name, key=page_key, use_container_width=True):
            st.session_state.page = page_name
            st.rerun()
        # Visual indicator using markdown (since button styling is limited)
        if st.session_state.page == page_name:
            st.markdown(f"<div style='height:2px; background:#4CAF50; margin:-8px 0 8px 0; width:80%; margin-left:10%;'></div>", unsafe_allow_html=True)

    st.markdown("<div class='nav-section'>Documentation</div>", unsafe_allow_html=True)

    doc_pages = [
        ("Methodology", "methodology"),
        ("Interpretation", "interpretation"),
    ]

    for page_name, page_key in doc_pages:
        if st.button(page_name, key=page_key, use_container_width=True):
            st.session_state.page = page_name
            st.rerun()
        if st.session_state.page == page_name:
            st.markdown(f"<div style='height:2px; background:#4CAF50; margin:-8px 0 8px 0; width:80%; margin-left:10%;'></div>", unsafe_allow_html=True)

    st.markdown("<div class='nav-section'>Presentation</div>", unsafe_allow_html=True)

    if st.button("Presentation Mode", key="presentation", use_container_width=True):
        st.session_state.page = "Presentation Mode"
        st.rerun()
    if st.session_state.page == "Presentation Mode":
        st.markdown(f"<div style='height:2px; background:#4CAF50; margin:-8px 0 8px 0; width:80%; margin-left:10%;'></div>", unsafe_allow_html=True)

    # Info box
    st.markdown("""
    <div class='info-box'>
        <p><strong>📊 Project Info</strong></p>
        <p>• Period: Nov 2024 - Nov 2025</p>
        <p>• Stocks: 2,545 companies</p>
        <p>• Data: FnGuide</p>
        <p>• Strategy: 20-day momentum</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    if summary_stats is not None:
        spread_ret = summary_stats.loc[2, 'Annual_Return_%']
        st.markdown(f"""
        <div style='background: rgba(231, 76, 60, 0.2); border-left: 4px solid #e74c3c; padding: 15px; border-radius: 5px; margin-top: 20px;'>
            <p style='color: white; margin: 0; font-size: 13px;'><strong>Key Finding:</strong></p>
            <p style='color: white; margin: 8px 0 0 0; font-size: 20px; font-weight: bold;'>{spread_ret:.1f}% Annual Spread</p>
            <p style='color: rgba(255, 255, 255, 0.8); margin: 5px 0 0 0; font-size: 12px;'>Reverse Momentum Detected</p>
        </div>
        """, unsafe_allow_html=True)

# Get current page
page = st.session_state.page

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🔄 Reverse Momentum in Korean Stocks</p>', unsafe_allow_html=True)

    st.markdown("""
    ## Does Momentum Work in Korean Market?

    **Research Question:** Can we profit by buying recent winners and selling recent losers?

    **Spoiler Alert:** We found the **OPPOSITE** - reverse momentum (mean reversion)!
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📊 Stocks Analyzed",
            value="2,545",
            delta="Korean Companies"
        )

    with col2:
        st.metric(
            label="📅 Trading Days",
            value="366",
            delta="Nov 2024 - Nov 2025"
        )

    with col3:
        st.metric(
            label="💾 Data Points",
            value="958,920",
            delta="High Quality Data"
        )

    st.markdown("---")

    # Quick summary
    if summary_stats is not None:
        st.subheader("⚡ Quick Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            **✅ What We Found:**
            - Recent LOSERS outperformed winners
            - Losers: +26.9% annually
            - Winners: +6.2% annually
            - Spread: -16.3% (negative = reverse momentum)
            """)

        with col2:
            st.info("""
            **📌 What This Means:**
            - Korean market shows mean reversion
            - Contrarian strategy works better
            - Different from US market behavior
            - Buy the dips, sell the rips!
            """)

    st.markdown("---")
    st.subheader("🎯 Navigate to:")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📈 Key Findings**\nSee the main results and charts")
    with col2:
        st.info("**📊 Detailed Results**\nDive into statistics")
    with col3:
        st.info("**🎤 Presentation Mode**\nFor your Dec 20 presentation")

# ============================================================================
# KEY FINDINGS
# ============================================================================
elif page == "Key Findings":
    st.markdown('<p class="main-header">📈 Key Findings</p>', unsafe_allow_html=True)

    if backtest_results is not None:
        # Main chart
        st.subheader("Winners vs Losers Cumulative Returns")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=backtest_results.index,
            y=backtest_results['winner_cumulative'],
            name='Winners (Top 10%)',
            line=dict(color='#2ecc71', width=3),
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=backtest_results.index,
            y=backtest_results['loser_cumulative'],
            name='Losers (Bottom 10%)',
            line=dict(color='#e74c3c', width=3),
            mode='lines'
        ))

        fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            title="Momentum Strategy: Winners vs Losers",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (1 = start)",
            hovermode='x unified',
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        st.subheader("📊 Performance Metrics")

        col1, col2, col3 = st.columns(3)

        winner_ret = summary_stats.loc[0, 'Annual_Return_%']
        loser_ret = summary_stats.loc[1, 'Annual_Return_%']
        spread_ret = summary_stats.loc[2, 'Annual_Return_%']

        with col1:
            st.metric(
                label="🏆 Winners Portfolio",
                value=f"{winner_ret:.1f}%",
                delta="Annual Return",
                delta_color="normal"
            )

        with col2:
            st.metric(
                label="📉 Losers Portfolio",
                value=f"{loser_ret:.1f}%",
                delta="Annual Return",
                delta_color="normal"
            )

        with col3:
            st.metric(
                label="🔄 Long-Short Spread",
                value=f"{spread_ret:.1f}%",
                delta="Reverse Momentum!",
                delta_color="inverse"
            )

        # Statistical significance
        st.markdown("---")
        st.subheader("📈 Statistical Test")

        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(backtest_results['long_short'], 0)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("T-Statistic", f"{t_stat:.2f}")
        with col2:
            st.metric("P-Value", f"{p_value:.4f}")
        with col3:
            if p_value < 0.05:
                st.success("✅ Statistically Significant")
            else:
                st.warning("⚠️ Not Significant (p > 0.05)")

        # Long-short returns distribution
        st.markdown("---")
        st.subheader("📊 Distribution of Long-Short Returns")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=backtest_results['long_short'],
            nbinsx=50,
            name='Long-Short Returns',
            marker_color='#9b59b6',
            opacity=0.7
        ))

        mean_return = backtest_results['long_short'].mean()
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero")
        fig.add_vline(x=mean_return, line_dash="dash", line_color="green",
                      annotation_text=f"Mean: {mean_return:.3f}%")

        fig.update_layout(
            title="Distribution of Daily Long-Short Returns",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DETAILED RESULTS
# ============================================================================
elif page == "Detailed Results":
    st.markdown('<p class="main-header">📊 Detailed Results</p>', unsafe_allow_html=True)

    if summary_stats is not None:
        st.subheader("Performance Summary Table")

        # Format the summary stats for display
        display_df = summary_stats.copy()
        display_df['Avg_Daily_Return_%'] = display_df['Avg_Daily_Return_%'].map('{:+.4f}%'.format)
        display_df['Annual_Return_%'] = display_df['Annual_Return_%'].map('{:+.2f}%'.format)
        display_df['Volatility_%'] = display_df['Volatility_%'].map('{:.4f}%'.format)
        display_df['Sharpe_Ratio'] = display_df['Sharpe_Ratio'].map('{:.2f}'.format)
        display_df['Win_Rate_%'] = display_df['Win_Rate_%'].map('{:.1f}%'.format)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Rolling performance
        st.subheader("📈 Rolling 30-Day Performance")

        rolling_30d = backtest_results['long_short'].rolling(30).mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rolling_30d.index,
            y=rolling_30d,
            name='30-Day Rolling Avg',
            line=dict(color='#ff7f0e', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.2)'
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.add_hline(y=backtest_results['long_short'].mean(), line_dash="dash",
                      line_color="green", opacity=0.7,
                      annotation_text=f"Overall Avg: {backtest_results['long_short'].mean():.3f}%")

        fig.update_layout(
            title="Rolling 30-Day Average Return (Long-Short)",
            xaxis_title="Date",
            yaxis_title="30-Day Avg Return (%)",
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Monthly performance
        st.markdown("---")
        st.subheader("📅 Monthly Returns Breakdown")

        monthly_returns = backtest_results.copy()
        monthly_returns['year_month'] = monthly_returns.index.to_period('M')
        monthly_agg = monthly_returns.groupby('year_month')['long_short'].agg([
            ('Mean', 'mean'),
            ('Std', 'std'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Count', 'count')
        ])
        monthly_agg.index = monthly_agg.index.astype(str)

        fig = go.Figure()

        colors = ['green' if x > 0 else 'red' for x in monthly_agg['Mean'].values]

        fig.add_trace(go.Bar(
            x=monthly_agg.index,
            y=monthly_agg['Mean'],
            name='Monthly Avg Return',
            marker_color=colors,
            opacity=0.7
        ))

        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

        fig.update_layout(
            title="Monthly Average Returns (Long-Short)",
            xaxis_title="Month",
            yaxis_title="Avg Return (%)",
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show table
        st.dataframe(monthly_agg.style.format({
            'Mean': '{:+.3f}%',
            'Std': '{:.3f}%',
            'Min': '{:+.3f}%',
            'Max': '{:+.3f}%'
        }), use_container_width=True)

# ============================================================================
# DATA EXPLORER
# ============================================================================
elif page == "Data Explorer":
    st.markdown('<p class="main-header">Data Explorer</p>', unsafe_allow_html=True)

    if stock_stats is not None:
        # Add sector classification, alpha, and beta
        stock_stats_with_sector = stock_stats.copy()
        stock_stats_with_sector['Sector'] = stock_stats_with_sector.index.map(classify_sector)

        # Add alpha and beta columns
        if stock_alphas is not None:
            stock_stats_with_sector['Alpha_%'] = stock_alphas
        else:
            stock_stats_with_sector['Alpha_%'] = np.nan

        if stock_betas is not None:
            stock_stats_with_sector['Beta'] = stock_betas
        else:
            stock_stats_with_sector['Beta'] = np.nan

        if stock_r_squared is not None:
            stock_stats_with_sector['R²'] = stock_r_squared
        else:
            stock_stats_with_sector['R²'] = np.nan

        # Sector filter
        st.subheader("Sector Filter")

        # Get sector counts
        sector_counts = stock_stats_with_sector['Sector'].value_counts()

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_sectors = st.multiselect(
                "Filter by sector (leave empty for all sectors):",
                options=sorted(sector_counts.index.tolist()),
                default=[]
            )
        with col2:
            st.markdown("#### Sector Distribution")
            st.write(f"Total: {len(stock_stats_with_sector)} stocks")
            for sector in sorted(sector_counts.index):
                st.write(f"{sector}: {sector_counts[sector]}")

        # Apply sector filter
        if selected_sectors:
            filtered_stocks = stock_stats_with_sector[stock_stats_with_sector['Sector'].isin(selected_sectors)]
            st.info(f"Showing {len(filtered_stocks)} stocks from {len(selected_sectors)} sector(s)")
        else:
            filtered_stocks = stock_stats_with_sector

        st.markdown("---")
        st.subheader("Top & Bottom Performers")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Top 10 Performers")
            top_10 = filtered_stocks.nlargest(10, 'Total_Return_%')
            display_cols = ['Sector', 'Total_Return_%', 'Alpha_%', 'Beta']
            st.dataframe(top_10[display_cols].style.format({
                'Total_Return_%': '{:+.2f}%',
                'Alpha_%': '{:+.2f}%',
                'Beta': '{:.2f}'
            }).background_gradient(subset=['Alpha_%'], cmap='RdYlGn', vmin=-20, vmax=20),
            use_container_width=True)

        with col2:
            st.markdown("### Bottom 10 Performers")
            bottom_10 = filtered_stocks.nsmallest(10, 'Total_Return_%')
            st.dataframe(bottom_10[display_cols].style.format({
                'Total_Return_%': '{:+.2f}%',
                'Alpha_%': '{:+.2f}%',
                'Beta': '{:.2f}'
            }).background_gradient(subset=['Alpha_%'], cmap='RdYlGn', vmin=-20, vmax=20),
            use_container_width=True)

        # Stock search
        st.markdown("---")
        st.subheader("Individual Stock Analysis")

        # Show sector in stock selection
        stock_options = sorted(filtered_stocks.index.tolist())
        stock_display = [f"{stock} ({filtered_stocks.loc[stock, 'Sector']})" for stock in stock_options]
        stock_mapping = dict(zip(stock_display, stock_options))

        selected_display = st.selectbox(
            "Select a stock:",
            options=stock_display
        )
        search_stock = stock_mapping.get(selected_display, stock_options[0] if stock_options else None)

        if search_stock:
            st.markdown(f"### {search_stock}")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total Return", f"{stock_stats.loc[search_stock, 'Total_Return_%']:+.2f}%")
            with col2:
                st.metric("Volatility", f"{stock_stats.loc[search_stock, 'Volatility_%']:.2f}%")
            with col3:
                # Display alpha
                if stock_alphas is not None and search_stock in stock_alphas.index:
                    alpha_value = stock_alphas[search_stock]
                    if not np.isnan(alpha_value):
                        st.metric("Alpha (α)", f"{alpha_value:+.2f}%")
                    else:
                        st.metric("Alpha (α)", "N/A")
                else:
                    st.metric("Alpha (α)", "N/A")
            with col4:
                # Display beta
                if stock_betas is not None and search_stock in stock_betas.index:
                    beta_value = stock_betas[search_stock]
                    if not np.isnan(beta_value):
                        st.metric("Beta (β)", f"{beta_value:.2f}")
                    else:
                        st.metric("Beta (β)", "N/A")
                else:
                    st.metric("Beta (β)", "N/A")
            with col5:
                # Display R²
                if stock_r_squared is not None and search_stock in stock_r_squared.index:
                    r2_value = stock_r_squared[search_stock]
                    if not np.isnan(r2_value):
                        st.metric("R² (fit)", f"{r2_value:.2f}")
                    else:
                        st.metric("R² (fit)", "N/A")
                else:
                    st.metric("R² (fit)", "N/A")

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Returns Chart", "Raw Data", "Statistics"])

            with tab1:
                if stock_prices is not None and search_stock in stock_prices.columns:
                    prices = stock_prices[search_stock]

                    fig = go.Figure()

                    # Price line
                    fig.add_trace(go.Scatter(
                        x=prices.index,
                        y=prices,
                        name='Price',
                        line=dict(color='#1f77b4', width=2),
                        mode='lines',
                        hovertemplate='%{x|%Y-%m-%d}<br>Price: ₩%{y:,.0f}<extra></extra>'
                    ))

                    # Add moving averages
                    ma_20 = prices.rolling(20).mean()
                    ma_60 = prices.rolling(60).mean()

                    fig.add_trace(go.Scatter(
                        x=ma_20.index,
                        y=ma_20,
                        name='20-Day MA',
                        line=dict(color='orange', width=1, dash='dash'),
                        hovertemplate='%{x|%Y-%m-%d}<br>20-MA: ₩%{y:,.0f}<extra></extra>'
                    ))

                    fig.add_trace(go.Scatter(
                        x=ma_60.index,
                        y=ma_60,
                        name='60-Day MA',
                        line=dict(color='red', width=1, dash='dash'),
                        hovertemplate='%{x|%Y-%m-%d}<br>60-MA: ₩%{y:,.0f}<extra></extra>'
                    ))

                    fig.update_layout(
                        title=f"{search_stock} - Price Chart with Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Price (KRW)",
                        height=500,
                        template="plotly_white",
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if stock_prices is not None and search_stock in stock_prices.columns:
                    # Calculate returns
                    prices = stock_prices[search_stock]
                    returns = prices.pct_change() * 100
                    cumulative_returns = (1 + returns/100).cumprod()

                    # Create subplot
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(f'{search_stock} - Daily Returns', 'Cumulative Return'),
                        vertical_spacing=0.12,
                        row_heights=[0.4, 0.6]
                    )

                    # Daily returns bar chart
                    colors = ['green' if x > 0 else 'red' for x in returns]
                    fig.add_trace(
                        go.Bar(
                            x=returns.index,
                            y=returns,
                            name='Daily Return',
                            marker_color=colors,
                            opacity=0.7,
                            hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:+.2f}%<extra></extra>'
                        ),
                        row=1, col=1
                    )

                    # Cumulative returns line
                    fig.add_trace(
                        go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns,
                            name='Cumulative Return',
                            line=dict(color='#1f77b4', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(31, 119, 180, 0.2)',
                            hovertemplate='%{x|%Y-%m-%d}<br>Cumulative: %{y:.2f}x<extra></extra>'
                        ),
                        row=2, col=1
                    )

                    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)

                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Daily Return (%)", row=1, col=1)
                    fig.update_yaxes(title_text="Cumulative Return (1 = start)", row=2, col=1)

                    fig.update_layout(
                        height=700,
                        template="plotly_white",
                        showlegend=False,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Return statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Return", f"{returns.mean():+.3f}%")
                    with col2:
                        st.metric("Median Return", f"{returns.median():+.3f}%")
                    with col3:
                        positive_days = (returns > 0).sum()
                        total_days = len(returns.dropna())
                        st.metric("Positive Days", f"{positive_days}/{total_days} ({positive_days/total_days*100:.1f}%)")
                    with col4:
                        st.metric("Std Dev", f"{returns.std():.3f}%")

            with tab3:
                if stock_prices is not None and search_stock in stock_prices.columns:
                    st.markdown("### 📋 Daily Price Data")

                    # Prepare data table
                    prices = stock_prices[search_stock]
                    returns = prices.pct_change() * 100

                    data_df = pd.DataFrame({
                        'Date': prices.index,
                        'Price (KRW)': prices.values,
                        'Daily Return (%)': returns.values,
                        'Change (KRW)': prices.diff().values
                    })

                    # Reverse order (newest first)
                    data_df = data_df.iloc[::-1].reset_index(drop=True)

                    # Show with formatting
                    st.dataframe(
                        data_df.style.format({
                            'Price (KRW)': '{:,.0f}',
                            'Daily Return (%)': '{:+.2f}',
                            'Change (KRW)': '{:+,.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )

                    # Download button
                    csv = data_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{search_stock}_data.csv",
                        mime="text/csv"
                    )

            with tab4:
                if stock_prices is not None and search_stock in stock_prices.columns:
                    st.markdown("### 📊 Detailed Statistics")

                    prices = stock_prices[search_stock]
                    returns = prices.pct_change() * 100

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Price Statistics**")
                        stats_df = pd.DataFrame({
                            'Metric': [
                                'Current Price',
                                'Starting Price',
                                'Highest Price',
                                'Lowest Price',
                                'Average Price',
                                'Median Price'
                            ],
                            'Value': [
                                f"₩{prices.iloc[-1]:,.0f}",
                                f"₩{prices.iloc[0]:,.0f}",
                                f"₩{prices.max():,.0f}",
                                f"₩{prices.min():,.0f}",
                                f"₩{prices.mean():,.0f}",
                                f"₩{prices.median():,.0f}"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("**Risk & Performance Metrics**")
                        returns_clean = returns.dropna()

                        # Get alpha, beta, and correlation
                        if stock_alphas is not None and search_stock in stock_alphas.index:
                            alpha_val = stock_alphas[search_stock]
                            alpha_str = f"{alpha_val:+.2f}%" if not np.isnan(alpha_val) else "N/A"
                        else:
                            alpha_str = "N/A"

                        if stock_betas is not None and search_stock in stock_betas.index:
                            beta_val = stock_betas[search_stock]
                            beta_str = f"{beta_val:.3f}" if not np.isnan(beta_val) else "N/A"
                        else:
                            beta_str = "N/A"

                        if stock_r_squared is not None and search_stock in stock_r_squared.index:
                            r2_val = stock_r_squared[search_stock]
                            r2_str = f"{r2_val:.3f}" if not np.isnan(r2_val) else "N/A"
                        else:
                            r2_str = "N/A"

                        # Calculate correlation with market
                        if market_returns is not None and beta_str != "N/A":
                            stock_ret = returns_clean
                            mkt_ret = market_returns.reindex(stock_ret.index).dropna()
                            common_idx = stock_ret.index.intersection(mkt_ret.index)
                            if len(common_idx) > 30:
                                corr = np.corrcoef(stock_ret[common_idx], mkt_ret[common_idx])[0, 1]
                                corr_str = f"{corr:.3f}"
                            else:
                                corr_str = "N/A"
                        else:
                            corr_str = "N/A"

                        stats_df = pd.DataFrame({
                            'Metric': [
                                'Total Return',
                                'Mean Daily Return',
                                'Volatility (Std Dev)',
                                '---',
                                'Alpha (α) Annualized',
                                'Beta (β)',
                                'R² (Model Fit)',
                                'Market Correlation',
                                '---',
                                'Best Day',
                                'Worst Day'
                            ],
                            'Value': [
                                f"{((prices.iloc[-1] / prices.iloc[0]) - 1) * 100:+.2f}%",
                                f"{returns_clean.mean():+.3f}%",
                                f"{returns_clean.std():.3f}%",
                                '',
                                alpha_str,
                                beta_str,
                                r2_str,
                                corr_str,
                                '',
                                f"{returns_clean.max():+.2f}%",
                                f"{returns_clean.min():+.2f}%"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)

                        # Alpha & Beta interpretation
                        col_a, col_b = st.columns(2)

                        with col_a:
                            if alpha_str != "N/A":
                                alpha_value = float(alpha_str.replace('%', '').replace('+', ''))
                                if alpha_value > 3:
                                    alpha_interpretation = "🌟 Excellent (Warren Buffett level)"
                                elif alpha_value > 1:
                                    alpha_interpretation = "✅ Good (skilled outperformance)"
                                elif alpha_value > -1:
                                    alpha_interpretation = "➖ Neutral (market-like)"
                                elif alpha_value > -3:
                                    alpha_interpretation = "⚠️ Poor (underperforming)"
                                else:
                                    alpha_interpretation = "❌ Bad (significant underperformance)"

                                st.success(f"**Alpha:** {alpha_interpretation}")

                        with col_b:
                            if beta_str != "N/A":
                                beta_value = float(beta_str)
                                if beta_value > 1.5:
                                    beta_interpretation = "🔴 Very Aggressive"
                                elif beta_value > 1.2:
                                    beta_interpretation = "🟠 Aggressive"
                                elif beta_value > 0.8:
                                    beta_interpretation = "🟡 Market-like"
                                elif beta_value > 0:
                                    beta_interpretation = "🟢 Defensive"
                                else:
                                    beta_interpretation = "🔵 Negative beta"

                                st.info(f"**Beta:** {beta_interpretation}")

                        # CAPM explanation
                        if alpha_str != "N/A" and beta_str != "N/A":
                            st.markdown("**📊 CAPM Model:**")
                            st.markdown(f"""
                            ```
                            Return = {alpha_str} + {beta_str} × Market Return

                            Alpha = Excess return (skill/luck)
                            Beta  = Market sensitivity (systematic risk)
                            R²    = {r2_str} (how well beta explains returns)
                            ```
                            """)

                            if r2_str != "N/A":
                                r2_value = float(r2_str)
                                if r2_value > 0.7:
                                    st.caption("✓ High R² means beta is a good predictor")
                                elif r2_value > 0.3:
                                    st.caption("⚡ Moderate R² means some idiosyncratic risk")
                                else:
                                    st.caption("⚠️ Low R² means stock has unique risk factors")

                    # Distribution histogram
                    st.markdown("**Return Distribution**")
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=returns_clean,
                        nbinsx=50,
                        name='Returns',
                        marker_color='#1f77b4',
                        opacity=0.7
                    ))
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    fig.add_vline(x=returns_clean.mean(), line_dash="dash", line_color="green",
                                  annotation_text=f"Mean: {returns_clean.mean():.2f}%")
                    fig.update_layout(
                        title="Distribution of Daily Returns",
                        xaxis_title="Daily Return (%)",
                        yaxis_title="Frequency",
                        height=300,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Multi-stock comparison
        st.markdown("---")
        st.subheader("Compare Multiple Stocks")

        selected_stocks = st.multiselect(
            "Select stocks to compare (up to 5):",
            options=sorted(filtered_stocks.index.tolist()),
            default=[filtered_stocks['Total_Return_%'].idxmax(),
                     filtered_stocks['Total_Return_%'].idxmin()] if len(filtered_stocks) >= 2 else [],
            max_selections=5
        )

        if selected_stocks and stock_prices is not None:
            st.markdown("### Price Comparison (Normalized to 100)")

            fig = go.Figure()

            for stock in selected_stocks:
                if stock in stock_prices.columns:
                    prices = stock_prices[stock]
                    normalized = (prices / prices.iloc[0]) * 100

                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized,
                        name=stock,
                        mode='lines',
                        line=dict(width=2),
                        hovertemplate=f'{stock}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>'
                    ))

            fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

            fig.update_layout(
                title="Stock Performance Comparison (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Normalized Price (100 = start)",
                height=500,
                template="plotly_white",
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Comparison table
            st.markdown("### Performance Comparison Table")
            comparison_df = stock_stats.loc[selected_stocks, [
                'Total_Return_%', 'Avg_Daily_Return_%', 'Volatility_%',
                'Max_Daily_Gain_%', 'Max_Daily_Loss_%'
            ]].copy()
            comparison_df.columns = ['Total Return', 'Avg Daily Return', 'Volatility',
                                     'Best Day', 'Worst Day']

            st.dataframe(
                comparison_df.style.format({
                    'Total Return': '{:+.2f}%',
                    'Avg Daily Return': '{:+.4f}%',
                    'Volatility': '{:.2f}%',
                    'Best Day': '{:+.2f}%',
                    'Worst Day': '{:+.2f}%'
                }).background_gradient(subset=['Total Return'], cmap='RdYlGn', vmin=-50, vmax=50),
                use_container_width=True
            )

        # Sector performance overview
        st.markdown("---")
        st.subheader("Sector Performance Overview")

        # Calculate sector averages
        sector_performance = filtered_stocks.groupby('Sector').agg({
            'Total_Return_%': 'mean',
            'Avg_Daily_Return_%': 'mean',
            'Volatility_%': 'mean'
        }).round(2)

        sector_counts = filtered_stocks['Sector'].value_counts()
        sector_performance['Count'] = sector_counts
        sector_performance = sector_performance.sort_values('Total_Return_%', ascending=False)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar chart of sector returns
            fig = go.Figure()
            colors = ['green' if x > 0 else 'red' for x in sector_performance['Total_Return_%']]

            fig.add_trace(go.Bar(
                x=sector_performance.index,
                y=sector_performance['Total_Return_%'],
                marker_color=colors,
                text=sector_performance['Total_Return_%'],
                texttemplate='%{text:+.1f}%',
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Return: %{y:+.2f}%<extra></extra>'
            ))

            fig.update_layout(
                title="Average Return by Sector",
                xaxis_title="Sector",
                yaxis_title="Avg Total Return (%)",
                height=400,
                template="plotly_white",
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Sector Statistics")
            st.dataframe(
                sector_performance.style.format({
                    'Total_Return_%': '{:+.2f}%',
                    'Avg_Daily_Return_%': '{:+.4f}%',
                    'Volatility_%': '{:.2f}%',
                    'Count': '{:.0f}'
                }).background_gradient(subset=['Total_Return_%'], cmap='RdYlGn', vmin=-50, vmax=50),
                use_container_width=True
            )

        # Scatter plot
        st.markdown("---")
        st.subheader("Return vs Volatility")

        fig = px.scatter(
            filtered_stocks.reset_index(),
            x='Volatility_%',
            y='Total_Return_%',
            hover_name=filtered_stocks.index,
            color='Sector',
            labels={
                'Volatility_%': 'Volatility (%)',
                'Total_Return_%': 'Total Return (%)'
            },
            title="Risk-Return Scatter Plot (Colored by Sector)"
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# METHODOLOGY
# ============================================================================
elif page == "Methodology":
    st.markdown('<p class="main-header">🎯 Methodology</p>', unsafe_allow_html=True)

    st.markdown("""
    ## Research Design

    ### 📋 Objective
    Test whether momentum investing (buying winners, selling losers) works in the Korean stock market.

    ### 📊 Data
    - **Source:** FnGuide
    - **Universe:** 2,545 Korean stocks (KOSPI + KOSDAQ)
    - **Period:** November 14, 2024 - November 14, 2025 (366 days)
    - **Quality:** 98.3% complete (1.7% missing data handled)

    ### 🔬 Strategy Implementation
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Strategy Parameters:**
        - **Lookback Period:** 20 trading days
        - **Winner Portfolio:** Top 10% by past return
        - **Loser Portfolio:** Bottom 10% by past return
        - **Rebalancing:** Daily
        - **Weighting:** Equal-weighted portfolios
        """)

    with col2:
        st.success("""
        **Daily Process:**
        1. Calculate 20-day return for each stock
        2. Rank all stocks by this return
        3. Form winner portfolio (top 10%)
        4. Form loser portfolio (bottom 10%)
        5. Measure next-day returns
        6. Repeat for 344 trading days
        """)

    st.markdown("""
    ### 📐 Mathematical Framework

    **Momentum Signal:**
    ```
    momentum(t) = (price(t) - price(t-20)) / price(t-20) × 100%
    ```

    **Portfolio Formation:**
    - **Winners:** stocks with momentum > 90th percentile
    - **Losers:** stocks with momentum < 10th percentile

    **Strategy Return:**
    ```
    Long-Short Return = Winner_Return - Loser_Return
    ```

    **Statistical Test:**
    - **Null Hypothesis (H₀):** Mean long-short return = 0 (no momentum)
    - **Alternative (H₁):** Mean long-short return ≠ 0 (momentum exists)
    - **Test:** Two-tailed t-test
    - **Significance level:** α = 0.05

    ### 📊 Performance Metrics

    **Return Metrics:**
    - Daily & annualized returns
    - Cumulative returns
    - Win rate (% profitable days)

    **Risk Metrics:**
    - Volatility (standard deviation)
    - Sharpe ratio = Return / Volatility × √252

    ### ⚠️ Limitations

    1. **Short time period:** Only 1 year of data
    2. **No transaction costs:** Real trading would reduce returns
    3. **Survivorship bias:** Only stocks that existed entire period
    4. **Market impact:** Assumes perfect execution
    5. **Short horizon:** 20-day lookback may be too short

    ### 🔬 Academic Foundation

    This study replicates:
    > **Jegadeesh, N., & Titman, S. (1993).** Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91.

    **Key difference:** Original study used 3-12 month horizons on US stocks. We used 20-day horizon on Korean stocks.
    """)

# ============================================================================
# INTERPRETATION
# ============================================================================
elif page == "Interpretation":
    st.markdown('<p class="main-header">💡 Interpretation</p>', unsafe_allow_html=True)

    st.markdown("""
    ## 🔄 What We Found: Reverse Momentum

    Instead of momentum (winners continue winning), we discovered **mean reversion** (losers bounce back).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.error("""
        ### ❌ Traditional Momentum
        **Expected (based on US studies):**
        - Winners outperform losers
        - Positive spread (Winners - Losers > 0)
        - Suggests market underreaction

        **Our Result:**
        - Did NOT find this in Korea
        """)

    with col2:
        st.success("""
        ### ✅ What We Actually Found
        **Reality in Korean Market:**
        - Losers outperform winners
        - Negative spread (Winners - Losers < 0)
        - Suggests market overreaction

        **Implication:**
        - Mean reversion, not momentum
        - Contrarian strategy works better
        """)

    st.markdown("---")

    st.subheader("🤔 Why Reverse Momentum?")

    st.markdown("""
    ### Possible Explanations:

    **1. 📱 Market Overreaction**
    - Korean market has high retail investor participation
    - Retail investors tend to overreact to news
    - Creates temporary mispricings that revert

    **2. 🕐 Time Horizon**
    - We used 20-day lookback (too short?)
    - US momentum studies use 3-12 months
    - Short-term = mean reversion, long-term = momentum?

    **3. 🌏 Market Structure**
    - Korean market is smaller than US
    - Less institutional participation
    - Higher volatility
    - Different market microstructure

    **4. 📊 Cultural Factors**
    - Research shows momentum weaker in Asian markets
    - Different investor psychology
    - Higher uncertainty avoidance

    **5. 📅 Time Period**
    - Nov 2024 - Nov 2025 may be unusual
    - Could be specific to this period
    - Need longer data to confirm

    ### 💼 Trading Implications
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **❌ Don't Do This:**
        - Buy recent winners
        - Chase performance
        - Follow momentum signals

        **Result:** Lost 16.3% annually
        """)

    with col2:
        st.success("""
        **✅ Do This Instead:**
        - Buy recent losers (contrarian)
        - Sell into strength
        - Fade extreme moves

        **Result:** Gained 26.9% annually
        """)

    st.markdown("""
    ---

    ### 🎓 Academic Contribution

    **What This Study Adds:**

    1. **First evidence** of reverse momentum in Korean stocks at 20-day horizon
    2. **Confirms** that momentum is not universal across markets
    3. **Suggests** market structure and culture matter
    4. **Practical** implications for Korean stock investors

    **Related Research:**
    - Chui, Titman, & Wei (2010): Momentum weaker in collectivist cultures
    - Griffin, Ji, & Martin (2003): Momentum varies across countries
    - Our finding: Supports cross-country differences

    ### ⚠️ Important Caveats

    1. **Not statistically significant** (p = 0.21 > 0.05)
       - Pattern exists but not conclusive
       - Need more data for confirmation

    2. **Transaction costs not included**
       - Real returns would be lower
       - High turnover = high costs

    3. **Short time period**
       - Only 1 year of data
       - Could be market-specific period
       - Need multi-year validation

    4. **Implementation challenges**
       - Daily rebalancing impractical
       - Market impact from trading
       - Slippage and commissions
    """)

# ============================================================================
# PRESENTATION MODE
# ============================================================================
elif page == "Presentation Mode":
    st.markdown('<p class="main-header">🎤 Presentation Mode</p>', unsafe_allow_html=True)

    # Slide selector
    slide = st.radio(
        "Select Slide:",
        ["Title", "Objective", "Data & Method", "Results", "Interpretation", "Conclusion"],
        horizontal=True
    )

    st.markdown("---")

    if slide == "Title":
        st.markdown("""
        # 🔄 Reverse Momentum in Korean Stocks

        ## Does Momentum Investing Work in Korea?

        <br><br>

        ### Research by: [Your Name]
        ### Seoul National University EMBA
        ### December 20, 2025

        <br>

        *Testing whether buying winners and selling losers generates profit*
        """, unsafe_allow_html=True)

    elif slide == "Objective":
        st.markdown("""
        ## 📋 Research Question

        ### Can we profit from momentum in Korean stocks?

        **Momentum Strategy:**
        - Buy recent winners (top 10%)
        - Sell recent losers (bottom 10%)
        - Hold and rebalance daily

        **Background:**
        - Jegadeesh & Titman (1993): Momentum works in US stocks
        - Returns 12% per year over 3-12 month horizon
        - Question: Does it work in Korea?

        **Hypothesis:**
        - H₀: No momentum (Winners = Losers)
        - H₁: Momentum exists (Winners > Losers)
        """)

    elif slide == "Data & Method":
        st.markdown("""
        ## 📊 Data & Methodology

        ### Data
        - **Source:** FnGuide
        - **Stocks:** 2,545 Korean companies
        - **Period:** Nov 2024 - Nov 2025 (366 days)
        - **Quality:** 98.3% complete

        ### Strategy
        1. Calculate 20-day past return for each stock
        2. Rank all stocks
        3. Buy top 10% (Winners)
        4. Sell bottom 10% (Losers)
        5. Measure next-day returns
        6. Repeat daily (344 days)

        ### Metrics
        - Daily & annual returns
        - Sharpe ratio
        - Statistical significance (t-test)
        """)

        # Show main chart
        if backtest_results is not None:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=backtest_results.index,
                y=backtest_results['winner_cumulative'],
                name='Winners',
                line=dict(color='#2ecc71', width=4)
            ))

            fig.add_trace(go.Scatter(
                x=backtest_results.index,
                y=backtest_results['loser_cumulative'],
                name='Losers',
                line=dict(color='#e74c3c', width=4)
            ))

            fig.update_layout(
                title="Winners vs Losers Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400,
                template="plotly_white",
                font=dict(size=16)
            )

            st.plotly_chart(fig, use_container_width=True)

    elif slide == "Results":
        st.markdown("""
        ## 📈 Results: Reverse Momentum!
        """)

        if summary_stats is not None:
            # Big metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                ### 🏆 Winners
                - **+6.2%** per year
                - Sharpe: 0.35
                - Win rate: 32%
                """)

            with col2:
                st.markdown("""
                ### 📉 Losers
                - **+26.9%** per year
                - Sharpe: 1.17
                - Win rate: 37.5%
                """)

            with col3:
                st.markdown("""
                ### 🔄 Spread
                - **-16.3%** per year
                - T-stat: -1.27
                - p-value: 0.21
                """)

            # Chart
            if backtest_results is not None:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['winner_cumulative'],
                    name='Winners (Top 10%)',
                    line=dict(color='#2ecc71', width=5),
                    mode='lines'
                ))

                fig.add_trace(go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['loser_cumulative'],
                    name='Losers (Bottom 10%)',
                    line=dict(color='#e74c3c', width=5),
                    mode='lines'
                ))

                fig.update_layout(
                    title="",
                    xaxis_title="",
                    yaxis_title="Cumulative Return",
                    height=500,
                    template="plotly_white",
                    font=dict(size=18),
                    showlegend=True,
                    legend=dict(font=dict(size=20))
                )

                st.plotly_chart(fig, use_container_width=True)

        st.error("### ❌ LOSERS BEAT WINNERS!")

    elif slide == "Interpretation":
        st.markdown("""
        ## 💡 Why Reverse Momentum?

        ### Possible Explanations:

        **1. Market Overreaction**
        - High retail participation
        - Emotional trading
        - Prices overshoot, then revert

        **2. Time Horizon**
        - 20 days may be too short
        - US studies use 3-12 months
        - Different dynamics at short horizons

        **3. Market Structure**
        - Korean ≠ US market
        - Smaller, higher volatility
        - Different investor behavior

        **4. Cultural Factors**
        - Asian markets show weaker momentum
        - Different risk preferences
        - Supported by academic research

        ### Trading Implication:

        **❌ Don't chase winners in Korea**
        **✅ Buy the dips (contrarian)**
        """)

    elif slide == "Conclusion":
        st.markdown("""
        ## 🎯 Conclusions

        ### What We Found:
        1. ❌ Momentum does NOT work in Korean market (20-day horizon)
        2. ✅ REVERSE momentum: Losers outperform (+26.9% vs +6.2%)
        3. 📊 Pattern exists but not statistically significant (p=0.21)
        4. 🌏 Korean market behaves differently from US

        ### Contributions:
        - First study of 20-day momentum in Korean stocks
        - Evidence that momentum is not universal
        - Practical insights for Korean investors

        ### Limitations:
        - Short time period (1 year)
        - No transaction costs
        - Need longer data for confirmation

        ### Future Research:
        - Test longer horizons (60-day, 120-day)
        - Multi-year analysis
        - Sector-specific momentum
        - Size effects

        ### Practical Takeaway:
        **In Korean market, buy the dips, sell the rips!**
        """)

        st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    <p>Korean Stock Market Momentum Analysis | Seoul National University EMBA</p>
    <p>Data: FnGuide | Period: Nov 2024 - Nov 2025 | Stocks: 2,545</p>
</div>
""", unsafe_allow_html=True)
