#!/bin/bash
# Install system dependencies for kitchen robot development

set -e

echo "🔧 Installing system dependencies..."

# Update package list
sudo apt-get update

# Install essential packages
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libeigen3-dev \
    libopencv-dev \
    libpcl-dev \
    docker.io \
    docker-compose

# Install ROS2 Humble if not already installed
if ! command -v ros2 &> /dev/null; then
    echo "📦 Installing ROS2 Humble..."
    sudo apt install software-properties-common
    sudo add-apt-repository universe
    sudo apt update && sudo apt install curl gnupg lsb-release
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
    sudo apt install ros-humble-desktop python3-colcon-common-extensions
fi

# Create Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements/development.txt

echo "✅ Dependencies installed successfully!"
echo "💡 Activate the virtual environment with: source venv/bin/activate"
