# Use the official Isaac Sim 5.0.0 container as the base image
FROM nvcr.io/nvidia/isaac-sim:5.0.0

# Switch to the root user to install system packages
USER root

# Update package lists and install required packages for Isaac Lab
RUN apt-get update && \
    apt-get install -y libopenh264-dev git cmake build-essential && \
    # Clean up the apt cache to reduce the final image size
    rm -rf /var/lib/apt/lists/*

# Set environment variables for Isaac Lab and terminal
ENV ISAAC_SIM_PATH=/isaac-sim
ENV TERM=xterm-256color

# Clone Isaac Lab to root directory (same level as isaac-sim)
RUN git clone https://github.com/isaac-sim/IsaacLab.git /IsaacLab

# Create symbolic link for Isaac Lab to find Isaac Sim
RUN cd /IsaacLab && ln -s /isaac-sim _isaac_sim

# Create wrapper scripts for easier Python access
RUN echo '#!/bin/bash\ncd /isaac-sim && ./python.sh "$@"' > /usr/local/bin/python && \
    chmod +x /usr/local/bin/python

RUN echo '#!/bin/bash\ncd /isaac-sim && ./python.sh "$@"' > /usr/local/bin/python3 && \
    chmod +x /usr/local/bin/python3

RUN echo '#!/bin/bash\ncd /isaac-sim && ./python.sh "$@"' > /usr/local/bin/isaac-python && \
    chmod +x /usr/local/bin/isaac-python

# Create a pip wrapper that uses Isaac Sim's Python
RUN echo '#!/bin/bash\ncd /isaac-sim && ./python.sh -m pip "$@"' > /usr/local/bin/pip && \
    chmod +x /usr/local/bin/pip

# Create aliases as backup (fixed the missing && here)
RUN echo 'alias python="/isaac-sim/python.sh"' >> /root/.bashrc && \
    echo 'alias pip="/isaac-sim/python.sh -m pip"' >> /root/.bashrc && \
    echo 'alias isaaclab="cd /IsaacLab && ./isaaclab.sh"' >> /root/.bashrc

# Install Isaac Lab dependencies (fixed the command and added TERM)
RUN cd /IsaacLab && TERM=xterm-256color ./isaaclab.sh -i

# Create Isaac Lab wrapper script
RUN echo '#!/bin/bash\ncd /IsaacLab && ./isaaclab.sh "$@"' > /usr/local/bin/isaaclab && \
    chmod +x /usr/local/bin/isaaclab

# Set the default working directory
WORKDIR /isaac-sim