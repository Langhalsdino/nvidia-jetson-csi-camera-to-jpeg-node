FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Set a working directory
WORKDIR /app

# Copy the project files into the container (adjust if needed)
COPY . .

ENTRYPOINT ["/make87/venv/bin/python3", "-m", "app.main"]  

# Default command, adjust as needed
CMD ["/app/main"]