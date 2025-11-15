# Python Financial Analysis - Complete Setup & Requirements Guide

## 📋 Table of Contents
1. [Software Installation](#1-software-installation)
2. [WRDS Account & Access Setup](#2-wrds-account--access-setup)
3. [Python Environment Setup](#3-python-environment-setup)
4. [Getting Your First Data](#4-getting-your-first-data)
5. [Understanding the Data](#5-understanding-the-data)
6. [Project Requirements](#6-project-requirements)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Software Installation

### ✅ What You Already Have
- [x] Python installed
- [x] WRDS module installed (`pip install wrds`)

### 🔧 What You Still Need

#### Install Anaconda (Recommended)
Anaconda gives you Python + all scientific libraries + Spyder IDE in one package.

**Download:** https://www.anaconda.com/download

**After installation, you should have:**
- Spyder (code editor)
- Anaconda Prompt (command line)
- Jupyter Notebook (optional, for interactive coding)

#### Required Python Libraries
Open **Anaconda Prompt** and run these commands:

```bash
# Core data analysis
pip install pandas numpy

# Database connection
pip install wrds sqlalchemy psycopg2-binary

# Visualization
pip install matplotlib seaborn

# Statistical analysis
pip install statsmodels scipy
```

#### Verify Installation
Open Spyder and run this test:

```python
import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
import statsmodels.api as sm

print("All libraries imported successfully!")
```

---

## 2. WRDS Account & Access Setup

### What is WRDS?
WRDS (Wharton Research Data Services) is a database containing financial data:
- Stock prices (CRSP)
- Company financials (Compustat)
- And much more

### 🎓 Getting WRDS Access

#### Step 1: Check if Your University Has WRDS
**Your university (SNU) likely has institutional access.**

1. Go to https://wrds-www.wharton.upenn.edu/
2. Look for "Register" or "Create Account"
3. **Use your university email** (e.g., @snu.ac.kr)
4. Follow registration process - you may need to:
   - Verify you're affiliated with SNU
   - Get professor approval (ask Professor Lee if needed)
   - Wait for account activation (can take 1-2 business days)

#### Step 2: Verify Your Account
Once you have credentials:
1. Go to https://wrds-www.wharton.upenn.edu/
2. Try logging in with your username/password
3. If successful, you should see the WRDS dashboard

#### Step 3: Set Up Python Connection

**Method 1: Manual Login (Easier for First Time)**

In Spyder, run:
```python
import wrds

# Connect to WRDS
db = wrds.Connection()
# You'll be prompted for:
# Username: <your_wrds_username>
# Password: <your_wrds_password>
```

**Method 2: Save Credentials (More Convenient)**

In Anaconda Prompt, run:
```bash
python
```
Then:
```python
import wrds
db = wrds.Connection()
# Enter username and password when prompted
# The system will ask if you want to save credentials - say YES
```

After this one-time setup, future connections won't ask for password.

#### Step 4: Test Your Connection

```python
import wrds

# Connect
db = wrds.Connection()

# List available libraries
db.list_libraries()
# You should see: ['audit', 'bank', 'block', 'bvd', ... 'crsp', 'comp', ...]

# List CRSP tables
db.list_tables(library="crsp")

# Close connection
db.close()
```

**✅ If this works, you have WRDS access!**

---

## 3. Python Environment Setup

### Recommended Folder Structure

Create this on your computer:

```
C:/Users/YourName/Finance_Project/
│
├── code/
│   ├── 01_data_download.py
│   ├── 02_data_cleaning.py
│   ├── 03_momentum_strategy.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── output/
│   ├── figures/
│   └── results/
│
└── docs/
    └── notes.md
```

### Create Working Directory Script

Save this as `setup_workspace.py`:

```python
import os

# Set your main project directory
project_dir = "C:/Users/YourName/Finance_Project"

# Create folder structure
folders = [
    "code",
    "data/raw",
    "data/processed", 
    "output/figures",
    "output/results",
    "docs"
]

for folder in folders:
    os.makedirs(os.path.join(project_dir, folder), exist_ok=True)
    
print(f"Project structure created at {project_dir}")
```

---

## 4. Getting Your First Data

### Understanding Data Sources

#### CRSP (Stock Prices)
- **Library:** `crsp`
- **Key Tables:**
  - `crsp.msf` = Monthly Stock File (monthly returns, prices)
  - `crsp.msi` = Monthly Stock Indices (market returns)
  - `crsp.stocknames` = Company identifiers

#### Compustat (Financial Statements)
- **Library:** `comp`
- **Key Tables:**
  - `comp.funda` = Fundamentals Annual (yearly financials)
  - `comp.fundq` = Fundamentals Quarterly

### Step-by-Step: Download Your First Dataset

Create `01_data_download.py`:

```python
import wrds
import pandas as pd

# Connect to WRDS
db = wrds.Connection()

# Query CRSP monthly stock data
# Start with just a few years to test
query = """
    SELECT date, permno, cusip, ret, prc, shrout
    FROM crsp.msf
    WHERE date BETWEEN '2020-01-01' AND '2024-12-31'
    AND ret IS NOT NULL
"""

# Execute query and get data
crsp_data = db.raw_sql(query)

# Save to CSV
crsp_data.to_csv("data/raw/crsp_monthly_2020_2024.csv", index=False)

print(f"Downloaded {len(crsp_data)} rows of CRSP data")
print(f"Date range: {crsp_data['date'].min()} to {crsp_data['date'].max()}")
print(f"Number of unique stocks: {crsp_data['permno'].nunique()}")

# Close connection
db.close()

# Preview the data
print("\nFirst few rows:")
print(crsp_data.head())
```

### Alternative: Download via WRDS Website

If Python connection doesn't work immediately:

1. Go to https://wrds-www.wharton.upenn.edu/
2. Log in
3. Navigate to: **Get Data → CRSP → Annual Update → Stock / Security Files → Monthly Stock File**
4. Select variables: `date, permno, cusip, ret, prc, shrout`
5. Set date range: 2020-01-01 to 2024-12-31
6. Download as CSV
7. Save to your `data/raw/` folder

---

## 5. Understanding the Data

### CRSP Variables Explained

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `date` | End of month date | 2024-01-31 |
| `permno` | Permanent company identifier | 10000 |
| `cusip` | Another company identifier | 68391610 |
| `ret` | Monthly return (decimal) | 0.05 (= 5%) |
| `prc` | Stock price at month-end | -45.25 |
| `shrout` | Shares outstanding (1000s) | 3680 |

**Important Notes:**
- `ret` = (Price_t - Price_{t-1} + Dividends) / Price_{t-1}
- Negative `prc` means it's an average of bid/ask (use absolute value)
- `shrout` is in thousands, so multiply by 1000 for actual shares
- Market cap = abs(prc) × shrout × 1000

### Compustat Variables Explained

| Variable | Description | Code in Compustat |
|----------|-------------|-------------------|
| Total Assets | Balance sheet total | `at` |
| Total Liabilities | All debts | `lt` |
| Stockholders' Equity | Book value | `seq` |
| Net Income | Profit/loss | `ni` |
| Sales Revenue | Total sales | `sale` |
| Interest Expense | Interest paid | `xint` |

### Quick Data Exploration Script

```python
import pandas as pd

# Load your data
df = pd.read_csv("data/raw/crsp_monthly_2020_2024.csv")

# Basic info
print("Dataset shape:", df.shape)
print("\nColumn names and types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())

# Check date range
df['date'] = pd.to_datetime(df['date'])
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

# How many unique stocks?
print(f"Unique stocks (permno): {df['permno'].nunique()}")

# Average monthly return
print(f"\nAverage monthly return: {df['ret'].mean():.4f} ({df['ret'].mean()*100:.2f}%)")
```

---

## 6. Project Requirements

### Your Team Project: Implement a Quant Strategy

#### Minimum Requirements

**1. Choose a Strategy**
- Momentum (buying winners, selling losers) - **recommended for replication**
- Value (low P/B ratio stocks)
- Size (small cap stocks)
- Quality (high ROE stocks)
- Or combine multiple factors

**2. Data Requirements**
- At least 10 years of data (e.g., 2000-2024)
- Minimum: CRSP monthly returns
- Better: CRSP + Compustat financials

**3. Analysis Steps**
```
Step 1: Download data from WRDS
Step 2: Clean and merge datasets
Step 3: Calculate signal (e.g., past 12-month return)
Step 4: Form portfolios (e.g., top/bottom 10%)
Step 5: Calculate portfolio returns
Step 6: Statistical testing (is it significant?)
Step 7: Visualize results
```

**4. Deliverables**
- Python code (well-commented)
- Final presentation showing:
  - Strategy description
  - Average returns
  - Statistical significance (t-stats)
  - Comparison to market
  - Visualizations

#### Recommended: Momentum Strategy Replication

**Goal:** Replicate Jegadeesh & Titman (1993)

**Pseudo-code:**
```python
# 1. Get CRSP data (1965-2024)
# 2. For each month t:
#    a. Calculate past 12-month return (month t-13 to t-2, skip t-1)
#    b. Rank stocks by past return
#    c. Form 10 portfolios (deciles)
#    d. Calculate next month (t+1) return for each portfolio
# 3. Average returns across all months
# 4. Test: Is (Top 10% return - Bottom 10% return) > 0?
```

#### Evaluation Criteria (Based on Syllabus)

**Team Presentation (500 points):**
- Clear explanation of strategy
- Correct implementation
- Statistical rigor
- Good visualizations
- Insights and discussion

---

## 7. Troubleshooting

### Common Issues & Solutions

#### ❌ "Can't connect to WRDS"
**Solutions:**
1. Check if you're on university network or VPN
2. Verify username/password at wrds-www.wharton.upenn.edu
3. Try: `pip install --upgrade wrds`
4. Email WRDS support: wrds@wharton.upenn.edu

#### ❌ "Module not found"
**Solution:**
```bash
pip install <module_name>
# or
conda install <module_name>
```

#### ❌ "Query takes forever"
**Solutions:**
1. Start with smaller date range (e.g., 5 years instead of 60)
2. Add `LIMIT 1000` to SQL query for testing
3. Download data once, save as CSV, then work locally

#### ❌ "Data looks weird / wrong"
**Checks:**
1. Look for missing values: `df.isnull().sum()`
2. Check for outliers: `df.describe()`
3. Verify date format: `pd.to_datetime(df['date'])`
4. Check for duplicates: `df.duplicated().sum()`

#### ❌ "My returns don't match the paper"
**Common mistakes:**
1. Forgot to exclude most recent month (t-1)
2. Used arithmetic mean instead of log returns for cumulative
3. Didn't handle missing data properly
4. Wrong portfolio formation (equal vs value-weighted)

---

## 🎯 Your Next Steps (Checklist)

### Week 1: Setup
- [ ] Install Anaconda
- [ ] Install required Python packages
- [ ] Register for WRDS account
- [ ] Test WRDS connection in Python
- [ ] Download small sample of CRSP data
- [ ] Create project folder structure

### Week 2: Learn Tools
- [ ] Practice pandas: merge, groupby, shift
- [ ] Understand CRSP data structure
- [ ] Calculate simple returns
- [ ] Make basic plots

### Week 3: Start Project
- [ ] Form team (if required)
- [ ] Choose strategy
- [ ] Download full dataset
- [ ] Start implementing portfolio formation

### Week 4-7: Execute & Refine
- [ ] Complete analysis
- [ ] Test statistical significance
- [ ] Create visualizations
- [ ] Prepare presentation

---

## 📚 Additional Resources

### Learning Python for Finance
- **Book:** "Python for Data Analysis" by Wes McKinney
- **Online:** DataCamp's "Finance Fundamentals in Python"
- **Documentation:** https://pandas.pydata.org/docs/

### Understanding Financial Data
- **WRDS Tutorial:** https://wrds-www.wharton.upenn.edu/pages/support/
- **CRSP Guide:** Available on WRDS website under "Research"
- **Paper:** Jegadeesh & Titman (1993) - read the methodology section

### Getting Help
1. **Professor:** kuanlee@snu.ac.kr
2. **TA:** thpaseong@snu.ac.kr  
3. **WRDS Support:** wrds@wharton.upenn.edu
4. **Stack Overflow:** For coding questions
5. **Classmates:** Form study groups

---

## 💡 Pro Tips

1. **Start Early:** WRDS access can take days to approve
2. **Test Small:** Always test code with small dataset first
3. **Save Often:** Save intermediate data to CSV files
4. **Comment Code:** Future you will thank present you
5. **Version Control:** Consider using Git/GitHub for team projects
6. **Ask for Help:** Don't struggle alone - use TA office hours

---

## ⚠️ Important Dates

- **Week 2 (Nov 8):** Python practical session
- **Week 8 (Dec 20):** Final team presentations
- **Attendance:** Miss 3+ classes = F grade

---

**Good luck! You've got this! 🚀**

If you get stuck on any step, refer back to this guide or reach out to the TA.