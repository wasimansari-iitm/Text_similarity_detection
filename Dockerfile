# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt first for Docker cache efficiency
COPY requirements.txt ./


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV FLASK_ENV=production

# Command to run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
