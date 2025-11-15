# 🚀 Factor Lab - Installation Guide

## Prerequisites

- **Python 3.8+** (check with `python3 --version`)
- **pip** (Python package manager)
- **Internet connection** (for downloading stock data)

---

## Step-by-Step Installation

### Method 1: Quick Start Script (Recommended)

```bash
# Navigate to project directory
cd quant1

# Run the automated setup script
./run.sh
```

**The script will automatically:**
1. Create a Python virtual environment
2. Install all dependencies
3. Initialize the SQLite database
4. Launch the Streamlit app

---

### Method 2: Manual Installation

#### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

#### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Packages that will be installed:**
- streamlit 1.27.0 (Web framework)
- pandas 2.1.0 (Data manipulation)
- numpy 1.24.0 (Numerical computing)
- yfinance 0.2.28 (Stock data)
- plotly 5.17.0 (Charts)
- scipy 1.11.0 (Statistics)
- scikit-learn 1.3.0 (Machine learning utilities)
- matplotlib 3.8.0 (Plotting)

**Installation time:** ~2-3 minutes

#### Step 3: Initialize Database

```bash
python3 setup.py
```

This will:
- Create `quant1_data.db` SQLite database
- Create all necessary tables (with `quant1_` prefix)
- Optionally test data download

**Expected output:**
```
============================================================
Setting up Factor Lab Database
============================================================

Created 8 tables:
  • quant1_backtest_results
  • quant1_backtest_runs
  • quant1_factors
  • quant1_fundamentals
  • quant1_performance
  • quant1_positions
  • quant1_prices
  • quant1_universe

✅ Setup Complete!
```

#### Step 4: Run the Application

```bash
streamlit run app.py
```

**The app will launch at:** `http://localhost:8501`

Your browser should open automatically. If not, manually navigate to the URL.

---

## Verification

### Test Core Modules

Run individual module tests to verify everything works:

```bash
# Test database
python3 core/database_schema.py

# Test data loader
python3 core/data_loader.py

# Test factors
python3 core/factors.py

# Test portfolio
python3 core/portfolio.py

# Test backtester
python3 core/backtest.py

# Test analytics
python3 core/analytics.py
```

Each test should end with: `✅ [Module name] test complete!`

---

## Troubleshooting

### Issue: "python3: command not found"

**Solution:**
```bash
# Try using 'python' instead
python --version

# If Python 3.8+, use:
python -m venv venv
```

### Issue: "pip: command not found"

**Solution:**
```bash
# Install pip
python3 -m ensurepip --upgrade

# Or download pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # (venv) should appear in prompt

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Data download fails

**Possible causes:**
1. **No internet connection** - Check your connection
2. **Yahoo Finance is down** - Try again later
3. **Rate limiting** - Use cached data or try fewer stocks

**Solutions:**
- Reduce date range (e.g., 1 year instead of 5)
- Use smaller stock universe (tech instead of sp500)
- Wait and retry (cached data will be used if available)

### Issue: Port 8501 already in use

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Or kill existing Streamlit process
pkill streamlit
```

### Issue: Database errors

**Solution:**
```bash
# Reset database
rm quant1_data.db

# Reinitialize
python3 setup.py
```

---

## First Time Usage

### Quick Test Strategy

1. **Homepage** → Click "Try Example Strategy"
2. **Strategy Builder** → Click "Run Backtest"
3. **Results** → View performance metrics

This will run a pre-configured momentum + value strategy on tech stocks.

**Expected results:**
- Total return: +30-50% (varies based on date range)
- Sharpe ratio: 0.8-1.5
- Max drawdown: -15% to -25%
- Backtest time: 30-60 seconds

---

## System Requirements

### Minimum
- **CPU**: 2 cores
- **RAM**: 4GB
- **Disk**: 500MB (for code + database)
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)

### Recommended
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Disk**: 2GB (with cached data)
- **Browser**: Chrome, Firefox, Safari (latest)

---

## Performance Tips

### Speed Up Backtests

1. **Use cached data** - Second run is much faster
2. **Reduce stocks** - Start with 7-10 stocks, then expand
3. **Shorter periods** - Test with 1-2 years first
4. **Less frequent rebalancing** - Quarterly instead of monthly

### Optimize Data Downloads

```python
# In data_loader.py, batch download:
loader.fetch_prices(tickers, start_date, end_date, threads=True)
```

---

## Deployment Options

### Local Network Access

Share with devices on your network:

```bash
streamlit run app.py --server.address=0.0.0.0
```

Access from other devices using: `http://YOUR_IP:8501`

Find your IP:
- **Mac/Linux**: `ifconfig | grep inet`
- **Windows**: `ipconfig`

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Click "Deploy"

**Free tier includes:**
- Unlimited public apps
- 1GB RAM
- Auto-deploy on git push
- Custom subdomain

---

## Updating the App

```bash
# Activate environment
source venv/bin/activate

# Pull latest changes (if using git)
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Run app
streamlit run app.py
```

---

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove database
rm quant1_data.db

# Remove project directory (if desired)
cd ..
rm -rf quant1
```

---

## Next Steps

After successful installation:

1. **Read the README.md** - Learn about features and architecture
2. **Try example strategies** - Homepage → "Try Example"
3. **Explore factors** - Factor Explorer page
4. **Build your own** - Strategy Builder
5. **Study the code** - Check `core/` modules
6. **Customize** - Add your own universes and factors

---

## Support

- **Documentation**: See `plan_files/` folder
- **Module docs**: Each Python file has detailed docstrings
- **Technical issues**: Run module tests to isolate problems

---

**Installation should take ~5-10 minutes total.**

**Happy backtesting! 🚀**
