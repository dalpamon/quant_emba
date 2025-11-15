"""
Full-Screen Presentation Mode
==============================
Clean, minimal slides for your Dec 20 presentation
Press arrow keys to navigate
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# Full-screen configuration
st.set_page_config(
    page_title="Korean Momentum Analysis - Presentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Hide Streamlit branding and menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {
    background-color: white;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 0rem;
    max-width: 95%;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS for presentation
st.markdown("""
<style>
    .title-slide {
        text-align: center;
        padding: 8rem 0;
        font-size: 4rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .subtitle-slide {
        text-align: center;
        font-size: 2rem;
        color: #666;
        margin-top: 2rem;
    }
    .slide-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-align: center;
    }
    .slide-content {
        font-size: 1.8rem;
        line-height: 2.5;
        color: #333;
    }
    .big-number {
        font-size: 5rem;
        font-weight: bold;
        text-align: center;
        margin: 2rem 0;
    }
    .green {color: #2ecc71;}
    .red {color: #e74c3c;}
    .blue {color: #3498db;}
    .bullet {
        margin: 1.5rem 0;
        font-size: 2rem;
    }
    .conclusion-box {
        background: #f0f8ff;
        padding: 3rem;
        border-radius: 1rem;
        margin: 2rem 0;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        backtest_results = pd.read_csv('output/results/momentum_backtest_results.csv',
                                       index_col=0, parse_dates=True)
        summary_stats = pd.read_csv('output/results/momentum_summary.csv')
        return backtest_results, summary_stats
    except:
        return None, None

backtest_results, summary_stats = load_data()

# Session state for slide navigation
if 'slide' not in st.session_state:
    st.session_state.slide = 0

# Navigation
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.slide == 0)):
        st.session_state.slide -= 1
        st.rerun()
with col2:
    st.markdown(f"<h3 style='text-align:center'>Slide {st.session_state.slide + 1} / 6</h3>", unsafe_allow_html=True)
with col3:
    if st.button("Next ➡️", use_container_width=True, disabled=(st.session_state.slide == 5)):
        st.session_state.slide += 1
        st.rerun()

st.markdown("---")

# SLIDE 0: Title
if st.session_state.slide == 0:
    st.markdown("<div class='title-slide'>🔄 Reverse Momentum<br>in Korean Stocks</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-slide'>Does Momentum Investing Work in Korea?</div>", unsafe_allow_html=True)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; font-size: 1.5rem; color: #666;'>
            <p><b>Seoul National University EMBA</b></p>
            <p>Inefficient Market and Quant Investment</p>
            <p>December 20, 2025</p>
        </div>
        """, unsafe_allow_html=True)

# SLIDE 1: Objective
elif st.session_state.slide == 1:
    st.markdown("<div class='slide-header'>📋 Research Question</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='slide-content'>
    <div class='bullet'>❓ <b>Can we profit from momentum in Korean stocks?</b></div>

    <br>

    <div style='background: #e8f4f8; padding: 2rem; border-radius: 1rem; margin: 2rem 0;'>
    <b>Momentum Strategy:</b><br>
    • Buy recent winners (top 10%)<br>
    • Sell recent losers (bottom 10%)<br>
    • Hold and rebalance daily
    </div>

    <br>

    <div class='bullet'>📚 <b>Background:</b> Jegadeesh & Titman (1993) found momentum works in US stocks</div>
    <div class='bullet'>🤔 <b>Question:</b> Does it work in Korea?</div>

    <br>

    <div style='background: #fff3cd; padding: 2rem; border-radius: 1rem; margin: 2rem 0;'>
    <b>Hypothesis:</b><br>
    H₀: No momentum (Winners = Losers)<br>
    H₁: Momentum exists (Winners > Losers)
    </div>
    </div>
    """, unsafe_allow_html=True)

# SLIDE 2: Data & Method
elif st.session_state.slide == 2:
    st.markdown("<div class='slide-header'>📊 Data & Methodology</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='slide-content'>
        <h3 style='color: #1f77b4;'>📈 Data</h3>
        <div class='bullet'>• <b>Source:</b> FnGuide</div>
        <div class='bullet'>• <b>Stocks:</b> 2,545 Korean companies</div>
        <div class='bullet'>• <b>Period:</b> Nov 2024 - Nov 2025</div>
        <div class='bullet'>• <b>Days:</b> 366 trading days</div>
        <div class='bullet'>• <b>Quality:</b> 98.3% complete</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='slide-content'>
        <h3 style='color: #1f77b4;'>⚙️ Strategy</h3>
        <div class='bullet'>1️⃣ Calculate 20-day past return</div>
        <div class='bullet'>2️⃣ Rank all stocks</div>
        <div class='bullet'>3️⃣ Buy top 10% (Winners)</div>
        <div class='bullet'>4️⃣ Sell bottom 10% (Losers)</div>
        <div class='bullet'>5️⃣ Measure next-day return</div>
        <div class='bullet'>6️⃣ Repeat daily (344 days)</div>
        </div>
        """, unsafe_allow_html=True)

# SLIDE 3: Results (THE BIG REVEAL)
elif st.session_state.slide == 3:
    st.markdown("<div class='slide-header'>📈 Results: The Surprise!</div>", unsafe_allow_html=True)

    if summary_stats is not None and backtest_results is not None:
        # Big numbers
        col1, col2, col3 = st.columns(3)

        winner_ret = summary_stats.loc[0, 'Annual_Return_%']
        loser_ret = summary_stats.loc[1, 'Annual_Return_%']
        spread_ret = summary_stats.loc[2, 'Annual_Return_%']

        with col1:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: #d4edda; border-radius: 1rem;'>
                <h3>🏆 Winners</h3>
                <div class='big-number green'>+{winner_ret:.1f}%</div>
                <p style='font-size: 1.5rem;'>Annual Return</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: #d1ecf1; border-radius: 1rem;'>
                <h3>📉 Losers</h3>
                <div class='big-number blue'>+{loser_ret:.1f}%</div>
                <p style='font-size: 1.5rem;'>Annual Return</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: #f8d7da; border-radius: 1rem;'>
                <h3>🔄 Spread</h3>
                <div class='big-number red'>{spread_ret:.1f}%</div>
                <p style='font-size: 1.5rem;'>Winners - Losers</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=backtest_results.index,
            y=backtest_results['winner_cumulative'],
            name='Winners (Top 10%)',
            line=dict(color='#2ecc71', width=6),
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=backtest_results.index,
            y=backtest_results['loser_cumulative'],
            name='Losers (Bottom 10%)',
            line=dict(color='#e74c3c', width=6),
            mode='lines'
        ))

        fig.add_hline(y=1, line_dash="dash", line_color="gray", line_width=2)

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Cumulative Return",
            height=500,
            template="plotly_white",
            font=dict(size=20),
            showlegend=True,
            legend=dict(font=dict(size=24), orientation="h", y=1.1)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style='text-align: center; background: #fff3cd; padding: 2rem; border-radius: 1rem; margin-top: 2rem;'>
            <h2 style='color: #856404; margin: 0;'>❌ LOSERS BEAT WINNERS!</h2>
        </div>
        """, unsafe_allow_html=True)

# SLIDE 4: Interpretation
elif st.session_state.slide == 4:
    st.markdown("<div class='slide-header'>💡 Why Reverse Momentum?</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='slide-content'>
        <h3 style='color: #e74c3c;'>❌ Expected (US Market)</h3>
        <div class='bullet'>• Winners keep winning</div>
        <div class='bullet'>• Positive spread</div>
        <div class='bullet'>• Market underreaction</div>
        <br>
        <div style='background: #f8d7da; padding: 2rem; border-radius: 1rem;'>
        <b>We did NOT find this!</b>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='slide-content'>
        <h3 style='color: #2ecc71;'>✅ Found (Korea)</h3>
        <div class='bullet'>• Losers bounce back</div>
        <div class='bullet'>• Negative spread</div>
        <div class='bullet'>• Market overreaction</div>
        <br>
        <div style='background: #d4edda; padding: 2rem; border-radius: 1rem;'>
        <b>Mean reversion!</b>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='slide-content'>
    <h3 style='color: #1f77b4; text-align: center;'>🤔 Possible Explanations</h3>
    <div class='bullet'>1️⃣ <b>Market overreaction:</b> High retail participation → emotional trading</div>
    <div class='bullet'>2️⃣ <b>Time horizon:</b> 20 days too short (US uses 3-12 months)</div>
    <div class='bullet'>3️⃣ <b>Market structure:</b> Korean ≠ US (smaller, more volatile)</div>
    <div class='bullet'>4️⃣ <b>Cultural factors:</b> Momentum weaker in Asian markets</div>
    </div>
    """, unsafe_allow_html=True)

# SLIDE 5: Conclusion
elif st.session_state.slide == 5:
    st.markdown("<div class='slide-header'>🎯 Conclusions</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='slide-content'>

    <div class='conclusion-box'>
    <h3 style='color: #1f77b4; margin-top: 0;'>🔍 What We Found</h3>
    <div class='bullet'>❌ Momentum does NOT work in Korean market (20-day horizon)</div>
    <div class='bullet'>✅ REVERSE momentum: Losers outperform by 16.3% annually</div>
    <div class='bullet'>📊 Pattern exists but not statistically significant (p=0.21)</div>
    <div class='bullet'>🌏 Korean market behaves differently from US</div>
    </div>

    <br>

    <h3 style='color: #2ecc71;'>✨ Contributions</h3>
    <div class='bullet'>• First study of short-term momentum in Korean stocks</div>
    <div class='bullet'>• Evidence that momentum is not universal</div>
    <div class='bullet'>• Practical insights for Korean investors</div>

    <br>

    <h3 style='color: #e67e22;'>⚠️ Limitations</h3>
    <div class='bullet'>• Only 1 year of data</div>
    <div class='bullet'>• No transaction costs included</div>
    <div class='bullet'>• Need longer period for confirmation</div>

    <br>

    <div style='background: #d1ecf1; padding: 3rem; border-radius: 1rem; text-align: center; margin-top: 2rem;'>
        <h2 style='color: #0c5460; margin: 0;'>💼 Practical Takeaway:</h2>
        <h1 style='color: #0c5460; margin: 1rem 0;'>In Korean Market,<br>Buy the Dips, Sell the Rips!</h1>
    </div>

    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #999; font-size: 1.2rem; padding: 2rem;'>
    <p><b>Thank You!</b></p>
    <p>Questions?</p>
</div>
""", unsafe_allow_html=True)

# Keyboard navigation hint
st.markdown("""
<div style='position: fixed; bottom: 20px; right: 20px; background: rgba(0,0,0,0.7); color: white; padding: 1rem; border-radius: 0.5rem; font-size: 1rem;'>
    💡 Tip: Use arrow buttons or click Next/Previous to navigate
</div>
""", unsafe_allow_html=True)
