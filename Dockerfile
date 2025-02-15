# Use official lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Expose the default Streamlit port (configurable)
EXPOSE 5052

# Run the application with a dynamic port
CMD ["streamlit", "run", "web_app/app.py", "--server.port=${PORT:-5052}", "--server.address=0.0.0.0"]
