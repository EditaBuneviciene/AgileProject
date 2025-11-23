# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application/main code
COPY main.py .
COPY test_main.py .

# Set environment variables
ENV PYTHONPATH=/app

# Run when container starts
CMD ["python", "main.py"]