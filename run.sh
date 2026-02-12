#!/bin/bash

# Run script for Semantic Search Engine
# CS-4015 Agentic AI - Homework 1 Phase 1

echo "========================================"
echo "Semantic Search Engine"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirement.txt

echo ""
echo "Starting application..."
echo ""

# Run the application
python app/main.py

# Deactivate virtual environment
deactivate
