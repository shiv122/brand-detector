# Linux Setup Guide

This guide helps you set up the Logo Detection API on Linux systems, especially when encountering issues with `uv` installation.

## Prerequisites

### 1. Python 3.11+
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip

# CentOS/RHEL
sudo yum install python3.11

# Check version
python3.11 --version
```

### 2. FFmpeg
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg

# Verify installation
ffmpeg -version
```

## Installing uv (Package Manager)

### Method 1: Automatic Installation (Recommended)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (choose one based on your shell)
source $HOME/.local/bin/env          # For bash/zsh
source $HOME/.local/bin/env.fish     # For fish shell

# Or manually add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uv --version
```

### Method 2: Manual PATH Addition
If the automatic installation doesn't work:

1. **Add to your shell profile**:
   ```bash
   # For bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   
   # For zsh
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

2. **Verify uv is available**:
   ```bash
   uv --version
   ```

### Method 3: Using pip as Fallback
If `uv` installation fails completely:

```bash
# Install dependencies with pip
pip install -r requirements.txt

# Or create a virtual environment first
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Troubleshooting uv Installation

### Issue: "Failed to install uv"

**Solution 1: Manual Installation**
```bash
# Download and install manually
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

**Solution 2: Check Permissions**
```bash
# Ensure you have write permissions
ls -la $HOME/.local/bin/
chmod +x $HOME/.local/bin/uv
```

**Solution 3: Use pip Instead**
```bash
# Skip uv and use pip directly
pip install -r requirements.txt
```

### Issue: "uv: command not found"

**Solution 1: Add to PATH**
```bash
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Solution 2: Create Symlink**
```bash
sudo ln -s $HOME/.local/bin/uv /usr/local/bin/uv
```

## Running the Application

### Option 1: Using the Startup Script
```bash
# Make executable
chmod +x start.sh

# Run with automatic setup
./start.sh

# Development mode
./start.sh --dev

# Custom port
./start.sh --port 8080
```

### Option 2: Manual Setup
```bash
# Install dependencies
uv sync  # or pip install -r requirements.txt

# Run the application
uv run python main.py  # or python main.py
```

### Option 3: Using pip (if uv fails)
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## Environment Setup

### Create Environment File
```bash
cat > .env << EOF
# Logo Detection API Configuration
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info
ENVIRONMENT=production

# Model Configuration
DEFAULT_WEIGHT=original.pt
CONFIDENCE_THRESHOLD=0.5
FRAMES_PER_SECOND=2

# File Paths
STATIC_DIR=static
FRAMES_DIR=static/frames
WEIGHTS_DIR=weights
EOF
```

### Create Required Directories
```bash
mkdir -p static/frames
mkdir -p static/temp_frames
mkdir -p weights
mkdir -p logs
```

## GPU Support on Linux

### NVIDIA GPU (CUDA)
```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only
```bash
# Install PyTorch for CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Testing

### Test GPU Acceleration
```bash
# With uv
uv run python test_gpu.py

# With pip
python test_gpu.py
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/api/health

# Device info
curl http://localhost:8000/api/device

# Weights info
curl http://localhost:8000/api/weights
```

## Common Issues and Solutions

### 1. Permission Denied
```bash
# Fix permissions
chmod +x start.sh
chmod +x $HOME/.local/bin/uv
```

### 2. Port Already in Use
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>

# Or use different port
./start.sh --port 8080
```

### 3. Missing Dependencies
```bash
# Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip ffmpeg

# Install Python dependencies
pip install -r requirements.txt
```

### 4. Virtual Environment Issues
```bash
# Create new virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Performance Optimization

### 1. GPU Memory
```bash
# Monitor GPU usage
nvidia-smi

# Clear GPU cache (if using CUDA)
python -c "import torch; torch.cuda.empty_cache()"
```

### 2. System Resources
```bash
# Monitor system resources
htop
free -h
df -h
```

### 3. Network Configuration
```bash
# Allow external connections
./start.sh --host 0.0.0.0 --port 8000
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `./start.sh --help` for script options
3. Check logs for error messages
4. Verify all prerequisites are installed
5. Test with a simple Python script first
