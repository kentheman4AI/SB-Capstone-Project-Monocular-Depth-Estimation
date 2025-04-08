# Use an official Python slim image as the base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port on which the Flask app runs (adjust if needed)
EXPOSE 5000

# Command to run the API server
CMD ["python", "DepthAnything_API.py"]