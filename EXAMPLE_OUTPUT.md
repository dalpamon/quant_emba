# Example: What Your Final Results Will Look Like

## Scenario: You Run Your Momentum Strategy

### The Process (Behind the Scenes)

**Month 1 (January 2000):**
```
You download data showing:
- Apple: +50% last year  → Rank #1   → Put in "Winners" portfolio
- Tesla: +30% last year  → Rank #150 → Put in "Winners" portfolio
- ...
- Kodak: -40% last year  → Rank #2500 → Put in "Losers" portfolio
- Blockbuster: -30%      → Rank #2499 → Put in "Losers" portfolio

Next month (Feb 2000):
- Winners portfolio: +2.1%
- Losers portfolio:  -0.5%
- Difference: +2.6%  ← You made money!
```

**Month 2 (February 2000):**
```
Recalculate everything:
- New rankings based on past 12 months
- Rebalance portfolios
- Measure returns for March 2000
```

**Repeat for 300 months (2000-2024)...**

---

## Your Final Results Would Look Like:

### Table 1: Summary Statistics
```
MOMENTUM STRATEGY PERFORMANCE (2000-2024)
==========================================

Portfolio            Avg Monthly    Annualized    Volatility    Sharpe    T-stat
                     Return         Return                      Ratio
------------------------------------------------------------------------------------
Winners (P10)        1.18%          15.12%        18.5%         0.82      4.12
Middle (P2-P9)       0.72%          8.98%         15.2%         0.59      2.87
Losers (P1)          0.21%          2.54%         22.3%         0.11      0.65
------------------------------------------------------------------------------------
Market (S&P 500)     0.68%          8.45%         16.1%         0.52      -
------------------------------------------------------------------------------------
Long-Short Spread    0.97%          12.58%        15.8%         0.79      4.45
(Winners - Losers)
                     ^^^^ THE MONEY-MAKER ^^^^

✅ T-stat = 4.45 (highly significant, need >2.0)
✅ Positive in 68% of months
✅ Works after transaction costs? Need to calculate...
```

**What this tells you:**
- If you bought winners and sold losers every month
- You'd make **12.58% per year** profit
- This is **4.13% better than the market**
- It's not just luck (T-stat = 4.45)

---

### Table 2: Returns by Year
```
YEAR-BY-YEAR BREAKDOWN
======================

Year    Winners    Losers    Spread    Market    Beat Market?
-------------------------------------------------------------
2000    +15.2%     -8.3%     +23.5%    -9.1%     ✅ Yes
2001    +8.1%      -12.4%    +20.5%    -11.9%    ✅ Yes
2002    -5.2%      -18.7%    +13.5%    -22.1%    ✅ Yes
2003    +42.3%     +28.1%    +14.2%    +28.7%    ✅ Yes
...
2020    +18.2%     -15.3%    +33.5%    +18.4%    ✅ Yes
2021    +25.1%     +8.2%     +16.9%    +28.7%    ❌ No
2022    -12.3%     -28.1%    +15.8%    -18.1%    ✅ Yes
2023    +32.1%     +18.4%    +13.7%    +26.3%    ✅ Yes
2024    +28.3%     +12.1%    +16.2%    +23.5%    ✅ Yes
-------------------------------------------------------------
Average +12.8%     -1.2%     +14.0%    +8.5%     ✅ 80% of years

⚠️ Note: 2021 was a reversal year (losers caught up)
```

---

### Chart 1: Cumulative Returns

```
Growth of $10,000 (2000-2024)

$200k |                                    ___---- Winners
      |                               ____/
$150k |                          ____/
      |                     ____/
$100k |                ____/  ___---- Market
      |           ____/   ___/
$50k  |      ____/   ___/
      | ____/   ___/
$10k  |/__---                    _---- Losers
      |_____________________----
      2000  2005  2010  2015  2020  2024

Final Values:
- Winners:  $185,430 (18.5x return!)
- Market:   $68,220  (6.8x return)
- Losers:   $12,450  (1.2x return)

Momentum Spread: $172,980 extra vs losers
```

---

### Chart 2: Monthly Distribution

```
Distribution of Monthly Returns
(Winners minus Losers)

Frequency
   |
40 |     ■■
   |     ■■■
30 |   ■■■■■
   |   ■■■■■
20 | ■■■■■■■■
   | ■■■■■■■■■
10 | ■■■■■■■■■■■
   | ■■■■■■■■■■■■
 0 |_______________
   -10% -5%  0%  5%  10% 15%

Mean: +0.97% per month
Median: +0.85%
% Positive: 68%
% > 2%: 35%
% < -2%: 12%

Interpretation:
- Most months are positive (good!)
- Some big wins (+15%)
- Few big losses (-10%)
```

---

### Chart 3: Rolling Performance

```
3-Year Rolling Returns
(Is momentum consistent?)

30% |  /\      /\         /\
    | /  \    /  \   /\  /  \
20% |/    \  /    \_/  \/    \
    |      \/
10% |                    /\
    |___________/\______/  \____
 0% |          /  \
    |_________________________
    2000    2008    2016    2024

Observations:
✅ Positive in 85% of 3-year periods
❌ Negative during 2008-2009 crisis
✅ Strongest during 2003-2007, 2020-2023
```

---

## Real Example: A Single Month in Detail

### January 2020 Snapshot

**Step 1: Rank stocks by past 12-month return**
```
Rank  Stock           Past 12m   Portfolio
                      Return     Assignment
================================================
1     TSLA (Tesla)    +149%      → Winners (Top 10%)
2     NVDA (Nvidia)   +108%      → Winners
3     AMD             +92%       → Winners
...
250   AAPL (Apple)    +35%       → Middle
...
2,480 XOM (Exxon)     -15%       → Losers (Bottom 10%)
2,481 GE              -18%       → Losers
2,482 F (Ford)        -22%       → Losers (Bottom 10%)
```

**Step 2: Measure February 2020 returns**
```
Portfolio         Return in Feb 2020
====================================
Winners (avg)     +2.4%
Middle (avg)      +1.1%
Losers (avg)      -0.8%
-----------------------------------
Spread            +3.2%  ← Money made this month!
```

**Step 3: Repeat every month for 24 years**

After 288 months, you average all the "Spreads":
- Average spread: **+0.97% per month**
- Annualized: **(1.0097)^12 - 1 = 12.3%**

---

## What You Present (5 Slides Example)

### Slide 1: The Question
**"Can Past Performance Predict Future Returns?"**
- Academic debate since 1960s
- Jegadeesh & Titman (1993): Yes, for 12 months
- We replicate their study with modern data (2000-2024)

### Slide 2: The Method
**Momentum Strategy Implementation**
1. Every month: Rank stocks by past 12-month return
2. Buy top 10% (Winners)
3. Sell bottom 10% (Losers)
4. Hold for 1 month, rebalance
5. Measure: Do Winners beat Losers?

### Slide 3: The Results
**[Show the cumulative returns chart]**
- Winners: 15.1% annual return
- Losers: 2.5% annual return
- Spread: 12.6% per year
- T-statistic: 4.45 (highly significant)

### Slide 4: When Does It Work?
**[Show year-by-year breakdown]**
- Works in 80% of years
- Fails during market reversals (2021)
- Strongest after crashes (2009, 2020)

### Slide 5: Conclusion
**Key Findings:**
✅ Momentum is real (not random)
✅ Generates 12.6% annual excess return
✅ But: High volatility, occasional crashes
❌ Transaction costs would reduce returns
🤔 Question: Why don't markets eliminate this?

---

## The "So What?" (Why This Matters)

### Academic Perspective
You're testing if markets are efficient:
- **Efficient market:** Past returns don't predict future
- **Your result:** They do! (Market "anomaly")

### Practical Perspective
You're showing hedge funds how they make money:
- Buy recent winners
- Short recent losers
- Pocket the spread

### Your Grade Perspective
You're demonstrating:
✅ Data analysis skills (Python, pandas)
✅ Financial knowledge (returns, portfolios, risk)
✅ Statistical rigor (hypothesis testing)
✅ Communication (presenting complex results clearly)

---

## Bottom Line

**Without WRDS:** You have nothing to analyze
**With WRDS:** You get 25 years of stock data
**After running code:** You get proof momentum works (or doesn't)
**In presentation:** You show "I tested this theory and here's what I found"

**Your output = Scientific proof of a trading strategy**

Not just "I think it works" → "Here's data showing it works"
