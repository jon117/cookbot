# Use the official Isaac Sim 5.0.0 container as the base image
FROM nvcr.io/nvidia/isaac-sim:5.0.0

# Switch to the root user to install system packages
USER root

# Update package lists and install the required video encoding library for WebRTC
RUN apt-get update && \
    apt-get install -y libopenh264-dev && \
    # Clean up the apt cache to reduce the final image size
    rm -rf /var/lib/apt/lists/*

# Set the default working directory
WORKDIR /isaac-sim