#!/bin/bash

# MovieSent Streamlit Local Test Script

echo "🎬 MovieSent - Starting Streamlit App"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Launching Streamlit app..."
echo "   Access at: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the server"
echo "======================================"
echo ""

# Run Streamlit
streamlit run streamlit_app.py
