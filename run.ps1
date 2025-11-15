# Factor Lab - Windows PowerShell Startup Script

Write-Host "🧪 Starting Factor Lab..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path "venv")) {
    Write-Host "⚠️  Virtual environment not found." -ForegroundColor Yellow
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        Write-Host "Make sure Python 3.8+ is installed" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Try running: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}

# Install dependencies if needed
if (-Not (Test-Path "venv\.installed")) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt

    if ($LASTEXITCODE -eq 0) {
        New-Item -Path "venv\.installed" -ItemType File -Force | Out-Null
    } else {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

# Check if database exists
if (-Not (Test-Path "quant1_data.db")) {
    Write-Host "🗄️  Database not found. Running setup..." -ForegroundColor Yellow
    python setup.py

    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Setup had issues, but continuing..." -ForegroundColor Yellow
    }
}

# Run Streamlit app
Write-Host ""
Write-Host "🚀 Launching Factor Lab..." -ForegroundColor Green
Write-Host "   App will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

streamlit run app.py
