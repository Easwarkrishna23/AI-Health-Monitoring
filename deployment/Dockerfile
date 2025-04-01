# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/ecg_model.h5 ./models/
COPY app/api.py .

# Expose port
EXPOSE 5000

# Run API
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]