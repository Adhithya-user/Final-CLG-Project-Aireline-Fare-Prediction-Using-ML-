# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install all the necessary Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code, models, and templates
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Start the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]