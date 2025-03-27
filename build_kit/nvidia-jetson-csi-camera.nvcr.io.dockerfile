FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Set a working directory
WORKDIR /app

# setup gstreamer libs
RUN apt-get update && \
    apt-get install libgstreamer1.0-dev gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0


# Copy the project files into the container (adjust if needed)
COPY . .

# Install venv
RUN python3 -m venv /make87/venv \
    && /make87/venv/bin/pip install --upgrade pip setuptools wheel \
    && /make87/venv/bin/pip install .

ENTRYPOINT ["python3", "-m", "app.main"]  

# Default command, adjust as needed
CMD ["/make87/venv/bin/python3/app/main"]