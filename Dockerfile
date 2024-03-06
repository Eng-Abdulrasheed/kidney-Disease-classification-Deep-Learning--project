FROM python:3.8-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file separately to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Set the command to run the application
CMD ["python3", "app.py"]
