# Stage 1: Build Stage – create the virtual environment and install dependencies
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3 AS base-image

ARG VIRTUAL_ENV=/opt/venv

# Install required system libraries including gstreamer dependencies
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends -y \
        python3.10-venv \
        libgstreamer1.0-dev \
        gir1.2-gstreamer-1.0 \
        gir1.2-gst-plugins-base-1.0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create a Python virtual environment and upgrade pip, setuptools, and wheel
# Since NVIDIA already provides python libs build for the jetson,
# this will make them available to the host system
RUN python3 -m venv --system-site-packages ${VIRTUAL_ENV} && \
    ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools wheel && \
    ${VIRTUAL_ENV}/make87/venv/bin/pip install .

# Copy the rest of your application code
COPY . .

# Use the virtual environment's python to run your application
ENTRYPOINT ["${VIRTUAL_ENV}/bin/python3", "-m", "app.main"]
