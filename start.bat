@echo off
setlocal enabledelayedexpansion

REM Logo Detection API - Complete Startup Script for Windows
REM This script handles installation, setup, and startup of the backend

echo 🚀 Logo Detection API - Complete Startup Script
echo ================================================

REM Parse command line arguments
set MODE=production
set PORT=8000

:parse_args
if "%1"=="" goto :main
if "%1"=="--dev" (
    set MODE=dev
    shift
    goto :parse_args
)
if "%1"=="--port" (
    set PORT=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--help" (
    echo Usage: %0 [--dev] [--port PORT]
    echo.
    echo Options:
    echo   --dev     Run in development mode with auto-reload
    echo   --port    Specify port (default: 8000)
    echo   --help    Show this help message
    exit /b 0
)
echo [ERROR] Unknown option: %1
echo Use --help for usage information
exit /b 1

:main
REM Check Python version
echo [INFO] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11+ is required
    exit /b 1
)
echo [SUCCESS] Python version check passed

REM Check uv
echo [INFO] Checking uv package manager...
uv --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] uv not found. Installing uv...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [ERROR] Failed to install uv
        echo Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/
        exit /b 1
    )
)
echo [SUCCESS] uv package manager found

REM Check FFmpeg
echo [INFO] Checking FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg not found
    echo Please install FFmpeg from https://ffmpeg.org/download.html
    echo Video processing may not work without FFmpeg
)

REM Setup environment
echo [INFO] Setting up environment...
if not exist "static\frames" mkdir "static\frames"
if not exist "static\temp_frames" mkdir "static\temp_frames"
if not exist "logs" mkdir "logs"

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo # Logo Detection API Configuration > .env
    echo PORT=8000 >> .env
    echo HOST=0.0.0.0 >> .env
    echo LOG_LEVEL=info >> .env
    echo ENVIRONMENT=production >> .env
    echo. >> .env
    echo # Model Configuration >> .env
    echo DEFAULT_WEIGHT=original.pt >> .env
    echo CONFIDENCE_THRESHOLD=0.5 >> .env
    echo FRAMES_PER_SECOND=2 >> .env
    echo. >> .env
    echo # File Paths >> .env
    echo STATIC_DIR=static >> .env
    echo FRAMES_DIR=static/frames >> .env
    echo WEIGHTS_DIR=weights >> .env
    echo [SUCCESS] Created .env file with default configuration
) else (
    echo [INFO] .env file already exists
)

REM Install dependencies
echo [INFO] Installing Python dependencies...
uv sync
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)
echo [SUCCESS] Dependencies installed successfully

REM Check weights
echo [INFO] Checking model weights...
if not exist "weights" (
    echo [WARNING] Weights directory not found. Creating...
    mkdir "weights"
)

dir /b "weights\*.pt" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] No model weights found in weights directory
    echo Please add your .pt model files to the weights directory
) else (
    echo [SUCCESS] Model weights found:
    dir /b "weights\*.pt"
)

REM Test GPU
echo [INFO] Testing GPU acceleration...
uv run python test_gpu.py
if errorlevel 1 (
    echo [WARNING] GPU acceleration test failed - will use CPU
) else (
    echo [SUCCESS] GPU acceleration test passed
)

REM Start server
echo [INFO] Starting Logo Detection API server...
echo [INFO] Mode: %MODE%
echo [INFO] Port: %PORT%
echo [INFO] API will be available at: http://localhost:%PORT%
echo [INFO] API documentation at: http://localhost:%PORT%/docs

if "%MODE%"=="dev" (
    uv run python main.py --dev --port %PORT%
) else (
    uv run python main.py --port %PORT%
)
