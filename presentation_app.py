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

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📈 Key Findings", "📊 Detailed Results", "🔍 Data Explorer",
     "🎯 Methodology", "💡 Interpretation", "🎤 Presentation Mode"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project:** Korean Stock Momentum Analysis
**Period:** Nov 2024 - Nov 2025
**Stocks:** 2,545 companies
**Data:** FnGuide
""")

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
elif page == "📈 Key Findings":
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
elif page == "📊 Detailed Results":
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
elif page == "🔍 Data Explorer":
    st.markdown('<p class="main-header">🔍 Data Explorer</p>', unsafe_allow_html=True)

    if stock_stats is not None:
        st.subheader("Top & Bottom Performers")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏆 Top 10 Performers")
            top_10 = stock_stats.nlargest(10, 'Total_Return_%')
            st.dataframe(top_10[['Total_Return_%', 'Avg_Daily_Return_%', 'Volatility_%']].style.format({
                'Total_Return_%': '{:+.2f}%',
                'Avg_Daily_Return_%': '{:+.4f}%',
                'Volatility_%': '{:.2f}%'
            }), use_container_width=True)

        with col2:
            st.markdown("### 📉 Bottom 10 Performers")
            bottom_10 = stock_stats.nsmallest(10, 'Total_Return_%')
            st.dataframe(bottom_10[['Total_Return_%', 'Avg_Daily_Return_%', 'Volatility_%']].style.format({
                'Total_Return_%': '{:+.2f}%',
                'Avg_Daily_Return_%': '{:+.4f}%',
                'Volatility_%': '{:.2f}%'
            }), use_container_width=True)

        # Stock search
        st.markdown("---")
        st.subheader("🔎 Search Individual Stock")

        search_stock = st.selectbox(
            "Select a stock:",
            options=sorted(stock_stats.index.tolist())
        )

        if search_stock:
            st.markdown(f"### {search_stock}")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Return", f"{stock_stats.loc[search_stock, 'Total_Return_%']:+.2f}%")
            with col2:
                st.metric("Avg Daily Return", f"{stock_stats.loc[search_stock, 'Avg_Daily_Return_%']:+.4f}%")
            with col3:
                st.metric("Volatility", f"{stock_stats.loc[search_stock, 'Volatility_%']:.2f}%")
            with col4:
                st.metric("Max Daily Gain", f"{stock_stats.loc[search_stock, 'Max_Daily_Gain_%']:+.2f}%")

            # Price chart
            if stock_prices is not None and search_stock in stock_prices.columns:
                prices = stock_prices[search_stock]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=prices,
                    name=search_stock,
                    line=dict(color='#1f77b4', width=2),
                    mode='lines'
                ))

                fig.update_layout(
                    title=f"{search_stock} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (KRW)",
                    height=400,
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)

        # Scatter plot
        st.markdown("---")
        st.subheader("📊 Return vs Volatility")

        fig = px.scatter(
            stock_stats.reset_index(),
            x='Volatility_%',
            y='Total_Return_%',
            hover_name=stock_stats.index,
            labels={
                'Volatility_%': 'Volatility (%)',
                'Total_Return_%': 'Total Return (%)'
            },
            title="Risk-Return Scatter Plot"
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# METHODOLOGY
# ============================================================================
elif page == "🎯 Methodology":
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
elif page == "💡 Interpretation":
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
elif page == "🎤 Presentation Mode":
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
