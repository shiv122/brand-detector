#!/bin/bash

# Logo Detection API - Complete Startup Script
# This script handles installation, setup, and startup of the backend

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        REQUIRED_VERSION="3.11"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
            print_success "Python $PYTHON_VERSION found (>= $REQUIRED_VERSION required)"
            return 0
        else
            print_error "Python $PYTHON_VERSION found, but $REQUIRED_VERSION+ is required"
            return 1
        fi
    else
        print_error "Python 3.11+ is required but not found"
        return 1
    fi
}

# Function to check and install uv
check_uv() {
    if command_exists uv; then
        print_success "uv package manager found"
        return 0
    else
        print_warning "uv not found. Installing uv..."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command_exists brew; then
                brew install uv
            else
                print_error "Homebrew not found. Please install Homebrew first: https://brew.sh"
                return 1
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            print_status "Installing uv on Linux..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            
            # Source the environment file to add uv to PATH
            if [ -f "$HOME/.local/bin/env" ]; then
                source "$HOME/.local/bin/env"
                print_success "Sourced uv environment"
            else
                print_warning "uv environment file not found, trying to add to PATH manually"
                export PATH="$HOME/.local/bin:$PATH"
            fi
            
            # Try to reload shell configuration
            if [ -f "$HOME/.bashrc" ]; then
                source "$HOME/.bashrc" 2>/dev/null
            elif [ -f "$HOME/.zshrc" ]; then
                source "$HOME/.zshrc" 2>/dev/null
            fi
        else
            print_error "Unsupported OS. Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
            return 1
        fi
        
        if command_exists uv; then
            print_success "uv installed successfully"
            return 0
        else
            print_warning "uv installation may have failed, trying alternative methods..."
            
            # Try to add to PATH manually
            export PATH="$HOME/.local/bin:$PATH"
            
            if command_exists uv; then
                print_success "uv found after manual PATH addition"
                return 0
            else
                print_warning "uv not found, will try to use pip as fallback"
                return 1
            fi
        fi
    fi
}

# Function to check FFmpeg
check_ffmpeg() {
    if command_exists ffmpeg; then
        print_success "FFmpeg found"
        return 0
    else
        print_warning "FFmpeg not found. Installing FFmpeg..."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command_exists brew; then
                brew install ffmpeg
            else
                print_error "Homebrew not found. Please install Homebrew first: https://brew.sh"
                return 1
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Ubuntu/Debian
            if command_exists apt; then
                sudo apt update && sudo apt install -y ffmpeg
            elif command_exists yum; then
                # CentOS/RHEL
                sudo yum install -y ffmpeg
            else
                print_error "Unsupported package manager. Please install FFmpeg manually"
                return 1
            fi
        else
            print_error "Unsupported OS. Please install FFmpeg manually: https://ffmpeg.org/download.html"
            return 1
        fi
        
        if command_exists ffmpeg; then
            print_success "FFmpeg installed successfully"
            return 0
        else
            print_error "Failed to install FFmpeg"
            return 1
        fi
    fi
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create necessary directories
    mkdir -p static/frames
    mkdir -p static/temp_frames
    mkdir -p logs
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
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
        print_success "Created .env file with default configuration"
    else
        print_status ".env file already exists"
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Try uv first
    if command_exists uv; then
        if uv sync; then
            print_success "Dependencies installed successfully with uv"
            return 0
        else
            print_warning "uv sync failed, trying pip as fallback"
        fi
    fi
    
    # Fallback to pip
    print_status "Using pip to install dependencies..."
    if pip install -r requirements.txt; then
        print_success "Dependencies installed successfully with pip"
        return 0
    else
        print_error "Failed to install dependencies with both uv and pip"
        return 1
    fi
}

# Function to check weights
check_weights() {
    print_status "Checking model weights..."
    
    if [ ! -d "weights" ]; then
        print_warning "Weights directory not found. Creating..."
        mkdir -p weights
    fi
    
    if [ -z "$(ls -A weights 2>/dev/null)" ]; then
        print_warning "No model weights found in weights directory"
        print_status "Please add your .pt model files to the weights directory"
        return 1
    else
        print_success "Model weights found:"
        ls -la weights/*.pt 2>/dev/null || print_warning "No .pt files found"
    fi
}

# Function to test GPU
test_gpu() {
    print_status "Testing GPU acceleration..."
    
    if command_exists uv; then
        if uv run python test_gpu.py; then
            print_success "GPU acceleration test passed"
        else
            print_warning "GPU acceleration test failed - will use CPU"
        fi
    else
        if python test_gpu.py; then
            print_success "GPU acceleration test passed"
        else
            print_warning "GPU acceleration test failed - will use CPU"
        fi
    fi
}

# Function to start the server
start_server() {
    local mode=${1:-production}
    local port=${2:-8000}
    
    print_status "Starting Logo Detection API server..."
    print_status "Mode: $mode"
    print_status "Port: $port"
    print_status "API will be available at: http://localhost:$port"
    print_status "API documentation at: http://localhost:$port/docs"
    
    if command_exists uv; then
        if [ "$mode" = "dev" ]; then
            uv run python main.py --dev --port "$port"
        else
            uv run python main.py --port "$port"
        fi
    else
        if [ "$mode" = "dev" ]; then
            python main.py --dev --port "$port"
        else
            python main.py --port "$port"
        fi
    fi
}

# Main execution
main() {
    echo "🚀 Logo Detection API - Complete Startup Script"
    echo "================================================"
    
    # Parse command line arguments
    MODE="production"
    PORT=8000
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                MODE="dev"
                shift
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [--dev] [--port PORT]"
                echo ""
                echo "Options:"
                echo "  --dev     Run in development mode with auto-reload"
                echo "  --port    Specify port (default: 8000)"
                echo "  --help    Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! check_python_version; then
        exit 1
    fi
    
    if ! check_uv; then
        exit 1
    fi
    
    if ! check_ffmpeg; then
        print_warning "FFmpeg installation failed - video processing may not work"
    fi
    
    # Setup environment
    setup_environment
    
    # Install dependencies
    if ! install_dependencies; then
        exit 1
    fi
    
    # Check weights
    if ! check_weights; then
        print_warning "No model weights found - server will start but detection won't work"
    fi
    
    # Test GPU
    test_gpu
    
    # Start server
    start_server "$MODE" "$PORT"
}

# Run main function with all arguments
main "$@"
