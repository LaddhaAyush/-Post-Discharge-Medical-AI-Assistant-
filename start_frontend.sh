#!/bin/bash

# Start script for Streamlit frontend
echo "Starting Post-Discharge AI Assistant Frontend..."

# Start Streamlit
exec streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true
