FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    
# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Create a non-root user and switch to it
RUN useradd -m landleaduser
USER landleaduser

EXPOSE 3100

CMD ["gunicorn", "app:app"]

# Command to run the application
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3100"]
 