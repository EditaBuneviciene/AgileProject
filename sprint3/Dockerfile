FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL application files
COPY *.py ./
COPY *.csv ./
COPY *.html ./
COPY *.js ./
COPY *.css ./

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port
EXPOSE 5000

# Start the API
CMD ["python", "api_connection.py"]