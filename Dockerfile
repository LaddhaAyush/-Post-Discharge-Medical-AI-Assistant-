# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Startup script to run both FastAPI and Streamlit
COPY docker_start.sh /docker_start.sh
RUN chmod +x /docker_start.sh

# Default command
CMD ["/docker_start.sh"] 