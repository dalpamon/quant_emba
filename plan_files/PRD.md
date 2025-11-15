# Product Requirements Document (PRD)
## Factor Lab - Multi-Factor Quantitative Investment Platform

**Version:** 1.0  
**Date:** October 25, 2024  
**Project Timeline:** October 25 - October 31, 2024 (Week 1 Sprint)  
**Product Owner:** You  
**Target Users:** Individual quant traders, students, finance professionals  

---

## 1. Executive Summary

### 1.1 Product Vision
Factor Lab is a web-based quantitative investment platform that empowers users to build, backtest, and analyze multi-factor trading strategies without writing code. It democratizes institutional-grade quant research tools for retail investors and students.

### 1.2 Problem Statement
- **Current Pain:** Aspiring quant traders must either use expensive platforms (Bloomberg, FactSet) or build everything from scratch
- **Market Gap:** No free, user-friendly tool exists that teaches factor investing while providing real backtesting capabilities
- **User Need:** A visual, mobile-friendly interface to experiment with factor strategies before risking real capital

### 1.3 Success Metrics
- **Learning Goal:** User understands 5+ quantitative factors by end of Week 1
- **Technical Goal:** Functional backtest engine with 3+ years of data
- **Usage Goal:** 10+ backtests run during development (self-testing)
- **Quality Goal:** Sharpe ratio calculations match academic standards

---

## 2. Product Objectives

### 2.1 Primary Objectives
1. **Educational First:** Teach factor investing fundamentals through interactive experimentation
2. **Research Tool:** Enable rapid strategy prototyping and testing
3. **Portfolio Readiness:** Build foundation for SNU course project (Nov-Dec)
4. **Commercial Viability:** Create MVP that can evolve into SaaS product (Project 3)

### 2.2 User Objectives

#### For You (Developer/Student):
- ✅ Master the quant workflow: Data → Factors → Portfolio → Backtest → Analyze
- ✅ Build portfolio piece demonstrating full-stack + quant skills
- ✅ Prepare for EMBA course with hands-on factor knowledge
- ✅ Create foundation for commercial product (Project 3)

#### For Future Users (Project 3):
- ✅ Test investment ideas without coding
- ✅ Learn which factors work in different market conditions
- ✅ Export strategies for paper trading
- ✅ Compare personal strategies against benchmarks

### 2.3 Non-Objectives (Out of Scope for Project 1)
- ❌ Real-time trading execution
- ❌ Machine learning models
- ❌ Social features (sharing strategies)
- ❌ Multi-user accounts / authentication
- ❌ Korean market data (save for Project 2)
- ❌ Advanced portfolio optimization (Markowitz, Black-Litterman)

---

## 3. User Personas

### Primary Persona: "Student Sam" (You)
- **Background:** EMBA student learning quantitative finance
- **Goal:** Prepare for course project, understand factor investing deeply
- **Pain Points:** Academic papers are theoretical, wants hands-on practice
- **Tech Savvy:** High (comfortable with Python, data science)
- **Usage Pattern:** Intensive during Week 1, reference tool during course

### Secondary Persona: "Curious Chris" (Future User)
- **Background:** Finance professional exploring quant strategies
- **Goal:** Backtest ideas from books/blogs before implementing
- **Pain Points:** Excel is tedious, doesn't want to learn Python
- **Tech Savvy:** Medium (can use web apps, not a coder)
- **Usage Pattern:** Weekly experimentation, monthly serious backtests

---

## 4. Core Features (MVP)

### 4.1 Data Management Module
**Priority:** P0 (Critical)

**Features:**
- Fetch historical stock data (US market via yfinance)
- Select stock universe (S&P 500, Tech stocks, Custom tickers)
- Date range selector (minimum 3 years)
- Data quality checks (missing dates, splits handling)

**Success Criteria:**
- Download 100+ stocks in <30 seconds
- Handle 5+ years of daily data without errors
- Auto-adjust for splits/dividends

---

### 4.2 Factor Library
**Priority:** P0 (Critical)

**Supported Factors:**

1. **Momentum** (12-month return, skip last month)
   - Formula: `(Price_t / Price_t-252) - 1`, exclude last 21 days
   - Academic Basis: Jegadeesh & Titman (1993)

2. **Value** (Book-to-Market ratio)
   - Formula: `1 / (Price / Book Value per Share)`
   - Academic Basis: Fama & French (1993)

3. **Size** (Market Capitalization)
   - Formula: `log(Market Cap)`
   - Academic Basis: Small-cap premium (Banz, 1981)

4. **Quality** (Profitability)
   - Formula: `Net Income / Total Assets` (ROA proxy)
   - Academic Basis: Novy-Marx (2013)

5. **Low Volatility** (Idiosyncratic Risk)
   - Formula: `Std Dev of returns (60-day rolling)`
   - Academic Basis: Low-vol anomaly (Ang et al., 2006)

**Factor Calculation Engine:**
- Z-score normalization for cross-sectional comparison
- Handle missing data (forward-fill, drop, or interpolate)
- Monthly recomputation for fundamental factors

**Success Criteria:**
- All 5 factors calculable in <5 seconds for 100 stocks
- Results match academic paper benchmarks (±2%)
- Clear documentation of each factor's logic

---

### 4.3 Strategy Builder
**Priority:** P0 (Critical)

**Features:**
- **Multi-Factor Combination:** Assign weights to each factor
  - UI: Sliders for each factor (0-100%)
  - Auto-normalize weights to sum to 100%
  - Composite score = weighted sum of z-scored factors

- **Portfolio Construction:**
  - **Long-Short:** Top 20% long, bottom 20% short
  - **Long-Only:** Top 20% long, rest cash
  - **Equal-Weight vs Value-Weight:** Position sizing options

- **Rebalancing Frequency:**
  - Monthly (default)
  - Quarterly
  - Annual

**Success Criteria:**
- User can change factor weights and see immediate impact
- Portfolio turnover calculated correctly
- Positions rebalance on correct dates (month-end)

---

### 4.4 Backtesting Engine
**Priority:** P0 (Critical)

**Features:**
- Simulate portfolio returns over selected time period
- Apply transaction costs (default: 10 bps per trade)
- Calculate daily equity curve
- Track positions over time

**Key Calculations:**
```
Daily Return = (Long Portfolio Return - Short Portfolio Return) / 2
Equity_t = Equity_t-1 × (1 + Return_t)
Transaction Cost = Turnover × Cost_per_Trade
```

**Success Criteria:**
- Backtest 100 stocks over 5 years in <10 seconds
- Handle corporate actions (splits, dividends) correctly
- No look-ahead bias (only use data available at decision time)

---

### 4.5 Performance Analytics
**Priority:** P0 (Critical)

**Metrics to Display:**

**Return Metrics:**
- Total Return (%)
- Annualized Return (CAGR)
- Monthly/Annual return distribution

**Risk Metrics:**
- Volatility (annualized std dev)
- Sharpe Ratio (risk-adjusted return)
- Max Drawdown (largest peak-to-trough decline)
- Sortino Ratio (downside risk focus)

**Factor Attribution:**
- Which factor contributed most to returns?
- Factor exposure over time
- Correlation between factor returns

**Visualization:**
- Equity curve (line chart)
- Drawdown chart (area chart)
- Monthly returns heatmap
- Factor contribution (bar chart)

**Success Criteria:**
- All metrics match academic formulas (verified against literature)
- Charts load in <2 seconds
- Mobile-friendly visualizations (responsive)

---

### 4.6 User Interface (Mobile-Responsive)
**Priority:** P0 (Critical)

**Key Screens:**
1. **Home/Dashboard** - Overview and quick start
2. **Strategy Builder** - Configure factors and portfolio
3. **Backtest Results** - Performance metrics and charts
4. **Factor Explorer** - Learn about each factor (educational)

**Design Principles:**
- **Mobile-first:** Touch-friendly controls, readable on 5" screens
- **Progressive disclosure:** Show simple options first, advanced later
- **Visual feedback:** Loading states, success/error messages
- **Educational:** Tooltips explaining each metric

**Success Criteria:**
- Usable on iPhone SE (smallest modern screen)
- All features accessible without horizontal scrolling
- Charts render correctly on mobile and desktop

---

## 5. Technical Requirements

### 5.1 Performance Requirements
- Data download: <30 seconds for 100 stocks
- Factor calculation: <5 seconds for 500 stocks
- Backtest execution: <10 seconds for 5-year period
- Page load: <3 seconds on 4G connection
- Mobile responsiveness: 60fps scrolling

### 5.2 Data Requirements
- Historical data: Minimum 3 years, target 10 years
- Update frequency: Daily (after market close)
- Data quality: 95%+ completeness (handle missing days)
- Storage: Local cache for faster re-runs

### 5.3 Browser Compatibility
- Chrome/Edge (latest 2 versions)
- Safari iOS (latest version)
- Firefox (latest 2 versions)
- Mobile web browsers (Chrome Android, Safari iOS)

### 5.4 Accessibility
- Keyboard navigation for all controls
- Screen reader compatible
- Color-blind friendly charts
- Minimum font size: 14px on mobile

---

## 6. User Flows

### 6.1 First-Time User Flow
1. Land on home page → See explanation of factor investing
2. Click "Try Example" → Pre-loaded momentum strategy runs
3. View results → Equity curve + metrics explained
4. Click "Build Your Own" → Go to Strategy Builder
5. Adjust factor weights → Run backtest
6. Compare to example → Understand what changed

### 6.2 Power User Flow (You)
1. Open app → Go directly to Strategy Builder
2. Select universe (Tech stocks) + date range
3. Set factor weights based on research paper
4. Run backtest → Analyze results
5. Tweak weights → Re-run (iterative testing)
6. Export results (screenshot or data) → Include in course notes

### 6.3 Learning Flow
1. Open "Factor Explorer" → Read about momentum
2. Click "See it in action" → Momentum-only backtest loads
3. View results → Understand momentum behavior
4. Repeat for other factors → Build mental models
5. Combine factors → Test multi-factor strategies

---

## 7. Wireframe References (See UI_ARCHITECTURE.md)

### Key Layout Patterns:
- **Desktop:** Sidebar (inputs) + Main panel (results)
- **Mobile:** Vertical stack, collapsible sections
- **Charts:** Full-width on mobile, side-by-side on desktop

---

## 8. Success Criteria & KPIs

### For Week 1 (Project 1):
- ✅ All 5 factors implemented and tested
- ✅ Backtest engine produces correct Sharpe ratios
- ✅ Mobile-responsive UI (tested on iOS/Android)
- ✅ 10+ successful backtests run (self-testing)
- ✅ Documentation complete (README, this PRD)

### For Course Project (Project 2):
- ✅ Korean market data integrated
- ✅ 3+ additional factors (from course papers)
- ✅ Team can use for final presentation
- ✅ Results match academic literature (validation)

### For Commercial Launch (Project 3):
- ✅ User authentication (login/signup)
- ✅ Save/load strategies
- ✅ 100+ registered users
- ✅ Positive user feedback (NPS > 40)

---

## 9. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Data source (yfinance) unreliable | High | Medium | Cache data locally; fallback to CSV import |
| Mobile performance issues | Medium | Medium | Lazy-load charts; pagination for large datasets |
| Calculation errors (bugs) | High | Low | Unit tests for all factors; verify against papers |
| Scope creep (too many features) | Medium | High | Stick to MVP; defer Project 2/3 features |
| Time constraint (1 week) | High | Medium | Use Streamlit for speed (migrate to React later if needed) |

---

## 10. Future Roadmap

### Project 2 (Nov-Dec, During Course):
- Korean market (KOSPI/KOSDAQ) via pykrx + FinanceDataReader
- Additional factors (earnings quality, liquidity)
- Multi-strategy comparison
- Regime detection (bull/bear markets)

### Project 3 (Jan 2026+, Commercial Product):
- User accounts & authentication
- Strategy marketplace (share/sell strategies)
- Paper trading integration (Alpaca, Interactive Brokers)
- Subscription model ($29/mo Pro tier)
- API access for developers
- Advanced analytics (factor timing, risk parity)

---

## 11. Appendix

### 11.1 Academic References
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers.
- Novy-Marx, R. (2013). The other side of value: The gross profitability premium.

### 11.2 Glossary
- **Factor:** A characteristic (momentum, value) that predicts stock returns
- **Long-Short:** Simultaneously buy and sell stocks to profit from relative performance
- **Sharpe Ratio:** Return per unit of risk (higher is better)
- **Alpha:** Excess return beyond what's explained by market risk
- **Backtest:** Simulating a strategy on historical data

### 11.3 Related Documents
- `DATA_ARCHITECTURE.md` - Database schema and data pipeline
- `UI_ARCHITECTURE.md` - Screen layouts and user flows
- `TECH_STACK.md` - Technology choices and justifications
- `IMPLEMENTATION_PLAN.md` - Day-by-day build schedule

---

**Document Status:** ✅ Final  
**Next Review:** After Project 1 completion (Oct 31)  
**Stakeholders:** You (developer/student), Future self (Project 3), Course team (Project 2)
