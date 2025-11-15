# UI Architecture Document
## Factor Lab - User Interface & Experience Design

**Version:** 1.0  
**Date:** October 25, 2024  
**Designer:** UX Architect  
**Platform:** Responsive Web (Desktop + Mobile)  

---

## 1. Design Philosophy

### 1.1 Core Principles

**1. Mobile-First**
- Design for 375px width first (iPhone SE)
- Progressive enhancement for larger screens
- Touch-friendly (44x44px minimum tap targets)

**2. Progressive Disclosure**
- Show simple options first
- Advanced features behind "Show More"
- Reduce cognitive load

**3. Educational Focus**
- Tooltips for every metric
- Inline help text
- Example strategies pre-loaded

**4. Performance**
- Lazy-load charts
- Skeleton screens during loading
- Optimistic UI updates

---

## 2. Information Architecture

### 2.1 Site Map

```
Home (Landing)
├── About Factor Investing
├── Try Example Strategy (Quick Start)
└── Get Started → Strategy Builder
    
Strategy Builder (Main App)
├── 1. Select Universe
│   ├── S&P 500
│   ├── Tech Stocks
│   └── Custom Tickers
│
├── 2. Configure Factors
│   ├── Factor Weights (Sliders)
│   ├── Factor Explorer (Learn More)
│   └── Save Preset
│
├── 3. Portfolio Settings
│   ├── Strategy Type (Long-Short / Long-Only)
│   ├── Rebalancing Frequency
│   └── Transaction Costs
│
└── 4. Run Backtest
    └── → Results Page

Results Dashboard
├── Summary Metrics (Cards)
├── Equity Curve (Chart)
├── Performance Analytics (Tabs)
│   ├── Returns
│   ├── Risk Metrics
│   ├── Factor Attribution
│   └── Holdings History
│
├── Export Results (JSON/CSV)
└── Modify & Re-run

Factor Explorer (Educational)
├── Factor Library
│   ├── Momentum
│   ├── Value
│   ├── Size
│   ├── Quality
│   └── Low Volatility
│
└── For each factor:
    ├── Definition
    ├── Academic Research
    ├── Historical Performance
    └── Try It → Pre-configured backtest
```

---

## 3. Screen Layouts

### 3.1 Home / Landing Page

**Purpose:** Explain value proposition, guide user to first action

**Layout (Desktop 1440px):**
```
┌─────────────────────────────────────────────────────────┐
│  [Logo: Factor Lab]              [Try Example] [Login]  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│         Hero Section                                     │
│    "Build & Test Quantitative                           │
│     Investment Strategies"                               │
│                                                          │
│    [Start Building] [Watch Demo]                         │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   How It Works (3 Cards)                                │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│   │ 1. Select│  │ 2. Build │  │3. Analyze│            │
│   │  Factors │  │ Strategy │  │ Results  │            │
│   └──────────┘  └──────────┘  └──────────┘            │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Pre-Built Examples                                     │
│   [Momentum Strategy] [Value Strategy] [Multi-Factor]   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Layout (Mobile 375px):**
```
┌──────────────────────┐
│ ☰  Factor Lab   [→]  │
├──────────────────────┤
│                      │
│  Build & Test        │
│  Quant Strategies    │
│                      │
│  [Start Building]    │
│  [Try Example]       │
│                      │
├──────────────────────┤
│  How It Works        │
│  ┌────────────────┐  │
│  │ 1. Select      │  │
│  │    Factors     │  │
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │ 2. Build       │  │
│  │    Strategy    │  │
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │ 3. Analyze     │  │
│  │    Results     │  │
│  └────────────────┘  │
│                      │
├──────────────────────┤
│  Examples            │
│  • Momentum          │
│  • Value             │
│  • Multi-Factor      │
└──────────────────────┘
```

---

### 3.2 Strategy Builder (Main App)

**Purpose:** Configure factors and portfolio settings

**Layout (Desktop):**
```
┌───────────────────────────────────────────────────────────────┐
│  Factor Lab          [Universe: S&P 500 ▼]  [Help] [Account] │
├───────────┬───────────────────────────────────────────────────┤
│           │                                                    │
│  Sidebar  │            Main Canvas                            │
│  (Config) │                                                    │
│           │                                                    │
│  ━━━━━━━━ │  Strategy Configuration                           │
│  Factors  │  ┌──────────────────────────────────────────────┐│
│  ━━━━━━━━ │  │ Momentum         [========○────] 40%         ││
│           │  │ Value            [=====○───────] 30%         ││
│  ☑ Momentum│ │ Quality          [=====○───────] 30%         ││
│    40%    │  │ Size             [────○─────────]  0%        ││
│           │  │ Volatility       [────○─────────]  0%        ││
│  ☑ Value  │  └──────────────────────────────────────────────┘│
│    30%    │                                                    │
│           │  ┌──────────────────────────────────────────────┐│
│  ☑ Quality│  │ Portfolio Settings                            ││
│    30%    │  │ • Type: Long-Short                            ││
│           │  │ • Rebalance: Monthly                          ││
│  ☐ Size   │  │ • Transaction Cost: 10 bps                    ││
│  ☐ Volatil│  └──────────────────────────────────────────────┘│
│           │                                                    │
│  ━━━━━━━━ │  Date Range: [2020-01-01] to [2024-12-31]        │
│  Settings │                                                    │
│  ━━━━━━━━ │                                                    │
│           │  [Run Backtest →]                                 │
│  Portfolio│                                                    │
│  Dates    │                                                    │
│  Costs    │                                                    │
│           │                                                    │
└───────────┴───────────────────────────────────────────────────┘
```

**Layout (Mobile - Vertical Stack):**
```
┌──────────────────────┐
│ ☰  Strategy Builder  │
├──────────────────────┤
│ Universe: S&P 500 ▼  │
├──────────────────────┤
│                      │
│ 📊 Factor Weights    │
│                      │
│ Momentum      40%    │
│ [========○────]      │
│                      │
│ Value         30%    │
│ [=====○───────]      │
│                      │
│ Quality       30%    │
│ [=====○───────]      │
│                      │
│ [+ Add Factor]       │
│                      │
├──────────────────────┤
│ ⚙️ Settings          │
│                      │
│ Type: Long-Short ▼   │
│ Rebalance: Monthly ▼ │
│ Cost: 10 bps         │
│                      │
├──────────────────────┤
│ 📅 Date Range        │
│ From: 2020-01-01     │
│ To:   2024-12-31     │
│                      │
├──────────────────────┤
│ [Run Backtest]       │
│                      │
└──────────────────────┘
```

---

### 3.3 Results Dashboard

**Purpose:** Display backtest results with metrics and visualizations

**Layout (Desktop):**
```
┌───────────────────────────────────────────────────────────────┐
│  Factor Lab    Momentum + Value Strategy      [Export] [Edit] │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  Key Metrics (Cards)                                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐│
│  │Total Return│ │    CAGR    │ │   Sharpe   │ │Max Drawdown││
│  │   +45.2%   │ │   +18.9%   │ │    1.45    │ │   -18.9%   ││
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘│
│                                                                │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  📈 Equity Curve                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                                           ╱             │  │
│  │                                    ╱────╱              │  │
│  │                          ╱────────╱                     │  │
│  │                 ╱───────╱                               │  │
│  │        ╱───────╱                                        │  │
│  │───────╱                                                 │  │
│  │ 2020   2021    2022    2023    2024                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  [Returns] [Risk] [Attribution] [Holdings]                    │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                │
│  Annual Returns                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 2020: +12.3%  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  │
│  │ 2021: +28.7%  ████████████████████████████░░░░░░░░░░░░ │  │
│  │ 2022: -15.2%  ░░░░░░░░░░░░░░░░                         │  │
│  │ 2023: +22.1%  ██████████████████████░░░░░░░░░░░░░░░░░░ │  │
│  │ 2024: +18.9%  ██████████████████░░░░░░░░░░░░░░░░░░░░░░ │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

**Layout (Mobile):**
```
┌──────────────────────┐
│ ← Momentum + Value   │
├──────────────────────┤
│ Total Return         │
│ +45.2%               │
│                      │
│ CAGR                 │
│ +18.9%               │
│                      │
│ Sharpe Ratio         │
│ 1.45                 │
│                      │
│ Max Drawdown         │
│ -18.9%               │
├──────────────────────┤
│ 📈 Equity Curve      │
│ ┌──────────────────┐ │
│ │       ╱          │ │
│ │    ╱─╱           │ │
│ │  ╱╱              │ │
│ │─╱                │ │
│ └──────────────────┘ │
│ [Expand Chart]       │
├──────────────────────┤
│ [Returns] [Risk]     │
│ [Attribution]        │
│                      │
│ Annual Returns       │
│ 2024: +18.9% ████    │
│ 2023: +22.1% █████   │
│ 2022: -15.2% ░       │
│                      │
├──────────────────────┤
│ [Export] [Edit]      │
└──────────────────────┘
```

---

### 3.4 Factor Explorer (Educational)

**Purpose:** Teach users about quantitative factors

**Layout (Desktop):**
```
┌───────────────────────────────────────────────────────────────┐
│  Factor Lab                         Factor Explorer           │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  Factor Library                                               │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 📊 Momentum                                       [Try→]│ │
│  │                                                           │ │
│  │ Definition: Stocks that have performed well in the       │ │
│  │ past 12 months tend to continue performing well.         │ │
│  │                                                           │ │
│  │ Academic Basis: Jegadeesh & Titman (1993)                │ │
│  │                                                           │ │
│  │ Historical Performance:                                   │ │
│  │ • Average Annual Return: +12.5%                           │ │
│  │ • Sharpe Ratio: 0.85                                      │ │
│  │ • Works best: Bull markets                                │ │
│  │                                                           │ │
│  │ [Learn More] [See Example Strategy]                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 💰 Value                                          [Try→]│ │
│  │                                                           │ │
│  │ Definition: Stocks with low price relative to book       │ │
│  │ value tend to outperform.                                 │ │
│  │ ...                                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. Component Library

### 4.1 Core Components

**1. Factor Weight Slider**
```
Component: FactorSlider
Props:
  - factorName: string
  - value: number (0-100)
  - onChange: (newValue) => void
  - tooltip: string

Visual:
┌────────────────────────────────┐
│ Momentum                    40%│
│ [========○────────────────]    │
│ ⓘ 12-month price momentum      │
└────────────────────────────────┘
```

**2. Metric Card**
```
Component: MetricCard
Props:
  - title: string
  - value: string/number
  - description: string
  - trend: 'up' | 'down' | 'neutral'

Visual:
┌─────────────────┐
│ Total Return    │
│   +45.2% ↑      │
│ Since 2020      │
└─────────────────┘
```

**3. Equity Curve Chart**
```
Component: EquityCurveChart
Props:
  - data: [{date, value}]
  - benchmark: [{date, value}] (optional)
  - height: number

Uses: Plotly.js for interactivity
Features:
  - Zoom/pan on desktop
  - Pinch-zoom on mobile
  - Hover tooltips
  - Drawdown overlay toggle
```

**4. Date Range Picker**
```
Component: DateRangePicker
Props:
  - startDate: Date
  - endDate: Date
  - onChange: (start, end) => void
  - minDate: Date (earliest available data)
  - maxDate: Date (today)

Visual (Mobile):
┌──────────────────────┐
│ From: [2020-01-01]   │
│ To:   [2024-12-31]   │
└──────────────────────┘
```

**5. Loading Skeleton**
```
Component: LoadingSkeleton
Used during: Data fetch, backtest execution

Visual:
┌────────────────────────────┐
│ ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░    │
│ Running backtest...        │
│ Calculating factors (2/5)  │
└────────────────────────────┘
```

---

## 5. User Flows

### 5.1 First-Time User Journey

**Goal:** Get user to first successful backtest in <5 minutes

```
Step 1: Landing Page
  ↓ Click "Try Example"
Step 2: Pre-loaded Strategy (Momentum)
  • Shows configured strategy
  • Auto-runs backtest
  • Displays results immediately
  ↓ "That was easy! Now build your own"
Step 3: Strategy Builder (Guided)
  • Inline tips: "Adjust these sliders to change factor weights"
  • Highlights: "Try increasing Value to 50%"
  • Live preview: "This will create a 60/40 Momentum/Value strategy"
  ↓ Click "Run Backtest"
Step 4: Results
  • Celebratory message: "Great! Your strategy returned +32.1%"
  • Comparison: "This outperformed Momentum-only by +5.3%"
  • Next steps: "Try adding Quality factor" or "Export results"
```

### 5.2 Power User Flow (You)

**Goal:** Rapid iteration on strategy ideas

```
Open App
  ↓ (Skips landing, goes directly to Strategy Builder)
Strategy Builder
  • Quick universe switch: Keyboard shortcut "U"
  • Preset factors: Saved configurations
  • Factor weights: Direct input (type "40" instead of sliding)
  ↓ Hit Enter (keyboard shortcut)
Results (loads in <3 seconds)
  • Quick scan: Sharpe, Max DD
  • Compare: Side-by-side with previous run
  ↓ Modify factor weights
Re-run (iterative loop)
  • 10-20 iterations in 15 minutes
  ↓ Found good strategy
Export
  • JSON: For further analysis in Python
  • PNG: For course presentation
  • CSV: For Excel analysis
```

### 5.3 Learning Flow

**Goal:** Understand each factor deeply

```
Factor Explorer Page
  ↓ Select "Momentum"
Factor Detail Page
  • Academic explanation
  • Visual intuition (charts)
  • Historical performance
  ↓ Click "Try Momentum Strategy"
Pre-configured Backtest Runs
  • Pure momentum (100% weight)
  • Shows equity curve, metrics
  • Explains: "Notice high volatility? That's why..."
  ↓ "How does Value compare?"
Switch to Value Factor
  • Shows Value-only backtest
  • Compare side-by-side
  ↓ "What if we combine them?"
Multi-Factor Strategy
  • 50/50 Momentum + Value
  • Shows diversification benefit
  • Learns: Factor combination reduces risk
```

---

## 6. Interaction Patterns

### 6.1 Desktop Interactions

**Hover States:**
- Metric cards: Expand to show detailed breakdown
- Chart points: Tooltip with date, value, return
- Factor names: Popup with definition

**Keyboard Shortcuts:**
```
Ctrl/Cmd + Enter : Run backtest
Ctrl/Cmd + E     : Export results
Ctrl/Cmd + S     : Save strategy
U                : Change universe
R                : Reset to defaults
?                : Show help/shortcuts
```

**Drag & Drop:**
- Reorder factors by priority
- Upload custom ticker CSV

### 6.2 Mobile Interactions

**Touch Gestures:**
- Swipe left/right: Navigate between result tabs
- Pinch-zoom: Charts
- Pull-down: Refresh data
- Long-press: Show context menu

**Bottom Sheets:**
- Settings panel slides up from bottom
- Smooth animation (300ms)
- Dismissible by swipe-down or tap outside

**Collapsible Sections:**
- Accordion for factor list
- "Show More" for advanced options
- Keeps screen uncluttered

---

## 7. Responsive Breakpoints

### 7.1 Breakpoint Strategy

```css
/* Mobile First */
/* Base styles: 375px (iPhone SE) */

@media (min-width: 640px) {
  /* sm: Larger phones, small tablets */
  /* - Increase font sizes
     - Side-by-side metric cards (2 columns) */
}

@media (min-width: 768px) {
  /* md: Tablets */
  /* - Show sidebar for navigation
     - 2-column layout for some sections */
}

@media (min-width: 1024px) {
  /* lg: Laptop */
  /* - Full sidebar + main panel layout
     - 4-column metric cards
     - Larger charts */
}

@media (min-width: 1280px) {
  /* xl: Desktop */
  /* - Max width: 1280px (centered)
     - More whitespace
     - Richer visualizations */
}
```

### 7.2 Adaptive Components

**Strategy Builder:**
- **Mobile:** Vertical stack, full-width sliders
- **Tablet:** 2-column (factors left, settings right)
- **Desktop:** Sidebar + main canvas

**Results Dashboard:**
- **Mobile:** Vertical stack, tabs for navigation
- **Tablet:** 2x2 grid for metric cards
- **Desktop:** 4-column cards + side-by-side charts

**Charts:**
- **Mobile:** Full-width, reduced height (250px)
- **Tablet:** Larger height (350px)
- **Desktop:** Even larger (450px), side-by-side layouts possible

---

## 8. Visual Design System

### 8.1 Color Palette

**Primary Colors:**
```
Brand Blue:    #00D9FF (buttons, accents)
Dark Blue:     #0A2540 (headers, text)
Success Green: #00C853 (positive returns)
Error Red:     #F44336 (negative returns, drawdowns)
Warning Orange:#FF9800 (alerts, warnings)
```

**Neutrals:**
```
Background:    #F8F9FA (light gray)
Surface:       #FFFFFF (cards, panels)
Border:        #E0E0E0 (dividers)
Text Primary:  #212121 (headings)
Text Secondary:#757575 (body, labels)
```

**Chart Colors:**
```
Equity Curve:  #00D9FF (primary line)
Benchmark:     #BDBDBD (comparison line)
Drawdown:      #FFCDD2 (shaded area)
```

### 8.2 Typography

**Font Stack:**
```css
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 
             'Segoe UI', Roboto, sans-serif;
```

**Scale:**
```
H1: 32px / 40px (Hero headings)
H2: 24px / 32px (Section headings)
H3: 20px / 28px (Card titles)
H4: 16px / 24px (Metric labels)
Body: 16px / 24px (Default text)
Small: 14px / 20px (Captions, help text)
```

**Mobile Adjustments:**
```
H1: 24px / 32px (smaller on mobile)
H2: 20px / 28px
Body: 16px / 24px (same as desktop)
Small: 14px / 20px (same as desktop)
```

### 8.3 Spacing System

**8px Grid:**
```
xs:  4px  (tight spacing)
sm:  8px  (default spacing)
md:  16px (card padding)
lg:  24px (section spacing)
xl:  32px (page margins)
xxl: 48px (hero sections)
```

### 8.4 Elevation (Shadows)

```css
/* Cards */
box-shadow: 0 2px 4px rgba(0,0,0,0.1);

/* Hover state */
box-shadow: 0 4px 8px rgba(0,0,0,0.15);

/* Modals */
box-shadow: 0 8px 16px rgba(0,0,0,0.2);
```

---

## 9. Accessibility (a11y)

### 9.1 WCAG 2.1 AA Compliance

**Color Contrast:**
- Text: 4.5:1 minimum ratio
- Large text: 3:1 minimum
- Interactive elements: 3:1 minimum

**Keyboard Navigation:**
- All features accessible via keyboard
- Visible focus indicators
- Logical tab order

**Screen Readers:**
- ARIA labels on all interactive elements
- Alt text for charts (data tables as fallback)
- Semantic HTML (header, nav, main, footer)

**Mobile Accessibility:**
- Minimum tap target: 44x44px (iOS guideline)
- No horizontal scrolling required
- Readable without zoom

### 9.2 Implementation Checklist

```html
<!-- Factor Slider -->
<div role="slider" 
     aria-label="Momentum factor weight"
     aria-valuemin="0"
     aria-valuemax="100"
     aria-valuenow="40"
     aria-valuetext="40 percent"
     tabindex="0">
  <input type="range" .../>
</div>

<!-- Metric Card -->
<div role="region" aria-label="Total return metric">
  <h3 id="total-return-label">Total Return</h3>
  <p aria-labelledby="total-return-label">
    <span aria-label="45.2 percent increase">+45.2%</span>
  </p>
</div>

<!-- Chart -->
<figure role="img" aria-label="Equity curve chart">
  <div id="chart"></div>
  <figcaption class="sr-only">
    Portfolio value grew from $100,000 to $145,200 over 5 years
  </figcaption>
</figure>
```

---

## 10. Animation & Transitions

### 10.1 Principles

**Purpose-Driven:**
- Show state changes (loading → loaded)
- Guide attention (new results appear)
- Provide feedback (button clicks)

**Performance:**
- Use CSS transforms (GPU-accelerated)
- Avoid animating layout properties
- 60fps target

**Durations:**
```
Fast:   150ms (hover states, ripples)
Medium: 300ms (panel transitions, modals)
Slow:   500ms (page transitions)
```

### 10.2 Key Animations

**1. Page Transitions**
```css
/* Fade in new page */
.page-enter {
  opacity: 0;
  transform: translateY(20px);
}
.page-enter-active {
  opacity: 1;
  transform: translateY(0);
  transition: all 300ms ease-out;
}
```

**2. Backtest Running**
```
Loading Skeleton:
┌────────────────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░ │ (animated shimmer)
│ Analyzing 485 stocks...    │
└────────────────────────────┘

Progress bar animates left to right
Text cycles through steps:
  "Downloading data..."
  "Calculating factors..."
  "Building portfolio..."
  "Running backtest..."
  "Analyzing results..."
```

**3. Chart Reveal**
```javascript
// Equity curve draws from left to right (500ms)
// Points fade in sequentially (stagger: 50ms)
// Final state: fully drawn chart with hover enabled
```

**4. Success Feedback**
```
After backtest completes:
┌────────────────────────────┐
│ ✅ Backtest Complete!      │ (bounce animation)
│ Your strategy returned...   │
└────────────────────────────┘
```

---

## 11. Error States & Empty States

### 11.1 Error Handling

**Data Fetch Error:**
```
┌────────────────────────────────┐
│ ⚠️ Couldn't load data          │
│                                │
│ Yahoo Finance seems slow.      │
│                                │
│ [Retry]  [Use Cached Data]     │
└────────────────────────────────┘
```

**Backtest Error:**
```
┌────────────────────────────────┐
│ ❌ Backtest failed             │
│                                │
│ Issue: Insufficient data       │
│ Ticker XYZZ not found          │
│                                │
│ [Remove Ticker]  [Get Help]    │
└────────────────────────────────┘
```

**Validation Error:**
```
┌────────────────────────────────┐
│ Factor weights must sum to 100%│
│ Current total: 87%             │
│                                │
│ [Auto-normalize]               │
└────────────────────────────────┘
```

### 11.2 Empty States

**No Saved Strategies:**
```
┌────────────────────────────────┐
│     📊                         │
│ No saved strategies yet        │
│                                │
│ Build your first strategy to   │
│ see it here.                   │
│                                │
│ [Start Building]               │
└────────────────────────────────┘
```

**No Results:**
```
┌────────────────────────────────┐
│     🔍                         │
│ Ready to backtest?             │
│                                │
│ Configure your strategy and    │
│ click "Run Backtest"           │
└────────────────────────────────┘
```

---

## 12. Performance Optimization

### 12.1 Loading Strategy

**Critical Rendering Path:**
1. Load HTML shell (< 1KB)
2. Load critical CSS inline
3. Show loading skeleton
4. Load JavaScript bundles (code-split)
5. Fetch data (parallel)
6. Render content

**Code Splitting:**
```javascript
// Load heavy chart library only when needed
const PlotlyChart = lazy(() => import('./PlotlyChart'));

// Load factor explorer only if user clicks
const FactorExplorer = lazy(() => import('./FactorExplorer'));
```

**Image Optimization:**
- Use WebP format
- Lazy-load images below fold
- Responsive images (srcset)

### 12.2 Chart Performance

**Plotly Optimization:**
```javascript
// For large datasets, downsample
if (dataPoints.length > 1000) {
  dataPoints = downsample(dataPoints, 500);
}

// Use scattergl for GPU acceleration
trace.type = 'scattergl';

// Disable unnecessary features on mobile
config = {
  displayModeBar: isMobile ? false : true,
  responsive: true
};
```

---

## 13. Mobile-Specific Patterns

### 13.1 Mobile Navigation

**Bottom Tab Bar:**
```
┌──────────────────────┐
│                      │
│   Main Content       │
│                      │
│                      │
├──────────────────────┤
│ [Builder] [Results]  │
│ [Factors] [Account]  │
└──────────────────────┘
```

**Hamburger Menu:**
```
☰ Menu
├─ Home
├─ Strategy Builder
├─ Results
├─ Factor Explorer
├─ Help
└─ Settings
```

### 13.2 Touch Interactions

**Pull-to-Refresh:**
- Pull down on results page to re-run backtest
- Visual feedback: spinner appears

**Swipe Gestures:**
- Swipe left/right: Navigate result tabs
- Swipe up: Dismiss bottom sheet
- Swipe down: Close modal

**Long Press:**
- Long-press metric card: Show detailed explanation
- Long-press factor: Quick edit weight

---

## 14. Implementation Notes

### 14.1 Recommended Tech Stack

**Frontend Framework:**
- **Option 1 (Quick MVP):** Streamlit
  - Pros: Fastest to build, Python-native
  - Cons: Limited mobile customization
  - Use for: Week 1 prototype

- **Option 2 (Production):** Next.js + React
  - Pros: Full control, best mobile support
  - Cons: Longer development time
  - Use for: Project 3 (commercial product)

**UI Library:**
- Tailwind CSS (utility-first, mobile-first)
- shadcn/ui (pre-built components)
- Radix UI (accessible primitives)

**Charts:**
- Plotly.js (interactive, mobile-friendly)
- Chart.js (lightweight alternative)

**State Management:**
- React Context (simple)
- Zustand (if complexity grows)

### 14.2 File Structure (React Version)

```
src/
├── components/
│   ├── FactorSlider.jsx
│   ├── MetricCard.jsx
│   ├── EquityCurveChart.jsx
│   ├── DateRangePicker.jsx
│   └── LoadingSkeleton.jsx
│
├── pages/
│   ├── HomePage.jsx
│   ├── StrategyBuilderPage.jsx
│   ├── ResultsPage.jsx
│   └── FactorExplorerPage.jsx
│
├── hooks/
│   ├── useBacktest.js
│   ├── useFactors.js
│   └── useResponsive.js
│
├── services/
│   ├── dataService.js
│   ├── backtestService.js
│   └── factorService.js
│
└── styles/
    ├── globals.css
    └── tailwind.css
```

---

## 15. User Testing Plan

### 15.1 Week 1 Testing (Self-Testing)

**Tasks:**
1. Build 5 different strategies
2. Test on iPhone (Safari) and Android (Chrome)
3. Test with slow network (3G throttling)
4. Test keyboard-only navigation

**Success Criteria:**
- All features work on mobile
- No horizontal scrolling
- Charts readable on small screen
- Backtest completes in <10 seconds

### 15.2 Project 2 Testing (Classmates)

**Tasks:**
1. Build and run a backtest (no instruction)
2. Find and understand a specific metric (Sharpe)
3. Compare two strategies
4. Export results

**Measure:**
- Time to complete each task
- Number of errors/confusion points
- User satisfaction (1-10 scale)

---

## 16. Appendix: Wireframe Assets

**Wireframe Notation:**
```
┌──────┐  Box (container)
│      │  
└──────┘

[Button]   Clickable button
[Text ▼]   Dropdown menu

━━━━━━━━━  Section divider

████████   Progress bar / filled state
░░░░░░░░   Empty state / loading skeleton

→  Next step / action arrow
✓  Success / completed state
⚠️  Warning / attention needed
❌ Error state
```

---

**Document Status:** ✅ Final  
**Next Steps:** Begin UI implementation (Day 4-5)  
**Owner:** You (Frontend Developer)  
**Related Docs:** `PRD.md`, `TECH_STACK.md`
