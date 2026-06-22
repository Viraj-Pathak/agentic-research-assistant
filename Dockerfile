FROM python:3.11-slim

# Security: run as non-root user
RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid 1000 --no-create-home appuser

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
