# Use a Python 3.8 slim image as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python requirements file into the container
COPY ./requirements.txt .

# Install Git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY ./exp ./exp
COPY ./models ./models
COPY ./utils ./utils
COPY ./data ./data
COPY ./run_cogflow_integration_informer_V2.py ./run_cogflow_integration_informer_V2.py

# Set the entry point to run the application
ENTRYPOINT ["python", "/app/run_cogflow_integration_informer_V2.py"]
CMD []