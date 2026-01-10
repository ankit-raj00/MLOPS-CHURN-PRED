FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (for pkg-config, etc. if needed)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy App Code
COPY . .

# Expose Port
EXPOSE 8000

# Run Application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
