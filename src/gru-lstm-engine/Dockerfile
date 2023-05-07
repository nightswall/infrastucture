# Use an official NVIDIA CUDA runtime as the base image
FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

# Install Python and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /gru-lstm

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the application dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

RUN chmod +x start.sh
# Expose the port that the application will run on
# EXPOSE 8000

# Set the environment variable for the Django settings module
ENV DJANGO_SETTINGS_MODULE myproject.settings


# Collect static files
# RUN python3.10 manage.py collectstatic --noinput
# RUN ls -ahl
# Start the application
CMD [ "bash","start.sh" ]

