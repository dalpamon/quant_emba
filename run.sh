#!/bin/bash

# Quick start script for Factor Lab

echo "🧪 Starting Factor Lab..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found."
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Check if database exists
if [ ! -f "quant1_data.db" ]; then
    echo "🗄️  Database not found. Running setup..."
    python3 setup.py
fi

# Run Streamlit app
echo ""
echo "🚀 Launching Factor Lab..."
echo "   App will open at: http://localhost:8501"
echo ""
streamlit run app.py
