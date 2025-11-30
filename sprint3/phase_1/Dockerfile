# Builder
FROM python:3.9-slim as builder

WORKDIR /app

# Copy Python dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytest pytest-cov beautifulsoup4 lxml

# Copy project source
COPY . .

# Test Image
FROM builder as test
CMD ["pytest", "-v", "--cov=main", "--cov-report=xml"]

# Production Image
FROM python:3.9-slim as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source files only
COPY main.py .
COPY api_connection.py .
COPY index.html .
COPY app.js .
COPY style.css .
COPY Data.csv .
COPY Data_Cleaned.csv .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Start API backend
CMD ["python", "api_connection.py"]
