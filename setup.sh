#!/bin/bash

echo "🎬 MovieSent Setup Script"
echo "========================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.12 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "⬇️  Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the application:"
echo "1. source .venv/bin/activate"
echo "2. python app_lstm.py"
echo "3. Open http://localhost:5000 in your browser"
echo ""
echo "🎬 Enjoy MovieSent!"