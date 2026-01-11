FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (for pkg-config, etc. if needed)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Production Requirements
COPY requirements_prod.txt .

# Install dependencies (Lighter, no dev tools)
RUN pip install --no-cache-dir -r requirements_prod.txt

# Set PYTHONPATH to ensure imports work correctly
ENV PYTHONPATH=/app

# Copy only necessary files
COPY params.yaml .
COPY setup.py .
COPY src/ src/
COPY app/ app/

# Expose Port
EXPOSE 8000

# Run Application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
