# Use an official Python runtime as a parent image
# Choose a version compatible with your dependencies (e.g., 3.10)
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
# --default-timeout=100 increases timeout for potentially large downloads
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the application code (assuming it's in an 'app' subfolder) into the container at /app
COPY ./app/ .

# Ensure data and analysis directories exist within the container image (optional, volumes handle runtime)
# RUN mkdir -p /app/data /app/analysis_files

# Set the default command to execute when the container starts
# This will run your main pipeline script
CMD ["python", "main.py"]
