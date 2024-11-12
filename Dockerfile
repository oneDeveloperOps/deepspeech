# Use the official Python 3.8.10 image from the Docker Hub
FROM python:3.8.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Install FFmpeg and wget
RUN apt-get update && apt-get install -y ffmpeg wget git && rm -rf /var/lib/apt/lists/*

RUN pip install flask[async]

RUN python3 -m pip install -U "yt-dlp[default]"

# Copy the requirements.txt file first to leverage Docker cache
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install git+https://github.com/openai/whisper.git 
# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
