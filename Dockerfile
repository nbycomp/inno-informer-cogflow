# Use a Python 3.8 slim image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /

# Copy the Python requirements file into the container
COPY ./requirements.txt .

# Install Git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application's code into the container
COPY ./exp /exp
COPY ./models /models
COPY ./utils /utils
COPY ./data /data
COPY ./run_cf_server.py .

# Set the entry point to run the application
ENTRYPOINT ["python", "run_cf_server.py"]


# docker build -t burntt/nby-cogflow-informer:latest .
# docker push burntt/nby-cogflow-informer:latest
