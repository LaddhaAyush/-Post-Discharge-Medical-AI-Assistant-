#!/bin/bash

# Start script for backend API
echo "Starting Post-Discharge AI Assistant Backend..."

# Create necessary directories
mkdir -p logs
mkdir -p data

# Start the FastAPI server
exec uvicorn backend_api:app --host 0.0.0.0 --port ${PORT:-8000}
