# Dockerfile for CI/CD artifacts

# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app

# Copy requirements
COPY sprint3/requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytest pytest-cov

# Copy all source code
COPY sprint3 .

# Stage 2: Test image
FROM builder as test
CMD ["python", "-m", "pytest", "test_main.py", "-v", "--cov=main", "--cov-report=xml"]

# Stage 3: Production image
FROM python:3.9-slim as production

WORKDIR /app

# Copy only the installed Python packages
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only production code
COPY sprint3/main.py .
COPY sprint3/Data.csv .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]