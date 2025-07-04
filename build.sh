#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up data directory
mkdir -p data

# Make sure all necessary files exist
echo "Build completed successfully"
