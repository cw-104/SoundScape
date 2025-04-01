#!/bin/bash

CLAD_VENV="CLAD/venv"
MAIN_VENV="venv"
REQ_FILE_CLAD="CLAD/requirements.txt"
REQ_FILE_MAIN="requirements.txt"

# Detect Python 3.9 dynamically
PYTHON_CMD=$(command -v python3.9 || command -v python3 || command -v python)

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python 3.9+ not found. Please install Python 3.9 or higher."
    exit 1
fi

echo " Using Python: $PYTHON_CMD"

# Check if the main venv exists
if [ ! -d "$MAIN_VENV" ]; then
    echo "Creating Main virtual environment..."
    "$PYTHON_CMD" -m venv "$MAIN_VENV"
    
    echo "Installing main dependencies..."
    "$MAIN_VENV/bin/pip" install --upgrade pip
    if [ -f "$REQ_FILE_MAIN" ]; then
        "$MAIN_VENV/bin/pip" install -r "$REQ_FILE_MAIN"
    else
        echo "⚠ Warning: Main requirements file not found. Skipping dependency installation."
    fi
else
    echo " Main virtual environment already exists. Skipping setup."
fi

# Check if CLAD venv exists
if [ ! -d "$CLAD_VENV" ]; then
    echo "Creating CLAD virtual environment..."
    "$PYTHON_CMD" -m venv "$CLAD_VENV"
    
    echo "Installing CLAD dependencies..."
    "$CLAD_VENV/bin/pip" install --upgrade pip
    if [ -f "$REQ_FILE_CLAD" ]; then
        "$CLAD_VENV/bin/pip" install -r "$REQ_FILE_CLAD"
    else
        echo "⚠ Warning: CLAD requirements file not found. Skipping dependency installation."
    fi
else
    echo " CLAD virtual environment already exists. Skipping setup."
fi

# Start API using the main venv
source "$MAIN_VENV/bin/activate"
echo "🚀 Starting API in main virtual environment..."
python3 src/start_api.py &

echo " Server is running and ready to accept requests!"
