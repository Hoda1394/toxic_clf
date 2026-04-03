#!/bin/bash

# Quick start script for toxic classifier

set -e

echo "========================================="
echo "Toxic Comment Classifier - Setup Script"
echo "========================================="

# Check Python version
echo "Python version:"
python --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing package..."
pip install -e .

# Run quick tests
echo ""
echo "Running quick tests..."
python tests/test_quick.py

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Prepare your data with 'text' and 'toxicity' columns"
echo "2. Run: python scripts/train.py --data_path data/train.csv"
echo "3. Or try the demo: python scripts/demo.py"
echo ""
echo "To deactivate virtual environment, run: deactivate"
