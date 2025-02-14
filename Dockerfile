# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    python3-pip \
    python3-dev \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PATH="/root/.cargo/bin:$PATH"

# Copy the requirements file
COPY requirements.txt .

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files
COPY . .

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["python", "web_app/app.py"]
# Expose port 10000
EXPOSE 10000


