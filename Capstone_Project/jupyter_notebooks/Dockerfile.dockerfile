# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (required for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and model files
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the application
# We default to app_v2.py. To run app.py, you can override the CMD at runtime or change it here.
ENTRYPOINT ["streamlit", "run", "EV_Battery_Health_Prediction_App_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]
