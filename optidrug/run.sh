#!/usr/bin/env bash

echo "==================================================="
echo "          OptiDrug Platform Launcher"
echo "==================================================="
echo ""

# Check for Python
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python is not installed. Please install Python 3.9+."
    exit 1
fi

echo "[1/3] Installing dependencies..."
$PY_CMD -m pip install -r requirements.txt

echo ""
echo "[2/3] Starting OptiDrug Server on http://localhost:5000 ..."
echo "[3/3] Open http://localhost:5000 in your browser."
echo "Press CTRL+C to stop the server."
echo ""

$PY_CMD server.py
