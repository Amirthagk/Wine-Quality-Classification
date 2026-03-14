FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Train the model during build (optional, but ensures it's ready in the image)
RUN python train.py

# Expose the API port
EXPOSE 8080

# Command to run the application using dynamically assigned PORT by Railway, defaulting to 8080
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"
