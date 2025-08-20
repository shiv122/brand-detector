# Logo Detection API Backend

A high-performance logo detection API built with FastAPI, YOLO, and GPU acceleration for Apple Silicon (M1/M2/M3) Macs.

## 🚀 Features

- **GPU Acceleration**: Automatic MPS (Metal Performance Shaders) support for Apple Silicon
- **Real-time Detection**: Fast logo detection in images and videos
- **Multiple Models**: Support for multiple YOLO model weights
- **Video Processing**: Frame-by-frame video analysis with FFmpeg
- **RESTful API**: Clean FastAPI endpoints with automatic documentation
- **Streaming Responses**: Real-time video processing with Server-Sent Events
- **Cross-platform**: Works on macOS, Linux, and Windows

## 📋 Requirements

### System Requirements
- **Python**: 3.11 or higher
- **FFmpeg**: For video processing
- **Apple Silicon Mac**: For GPU acceleration (M1/M2/M3)

### Dependencies
- FastAPI 0.116.1+
- Uvicorn 0.35.0+
- Ultralytics 8.3.170+
- PyTorch 2.1.0+ (with MPS support)
- OpenCV 4.12.0+
- Python-multipart 0.0.20+

## 🛠️ Quick Start

### Option 1: Automatic Setup (Recommended)

#### macOS/Linux
```bash
# Make script executable
chmod +x start.sh

# Start the server (production mode)
./start.sh

# Start in development mode with auto-reload
./start.sh --dev

# Start on custom port
./start.sh --port 8080
```

#### Windows
```cmd
# Start the server (production mode)
start.bat

# Start in development mode
start.bat --dev

# Start on custom port
start.bat --port 8080
```

### Option 2: Manual Setup

1. **Install Python 3.11+**
   ```bash
   # macOS
   brew install python@3.11
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install python3.11
   ```

2. **Install uv package manager**
   ```bash
   # macOS
   brew install uv
   
   # Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install FFmpeg**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   ```

4. **Clone and setup the project**
   ```bash
   cd backend
   uv sync
   ```

5. **Add model weights**
   ```bash
   # Create weights directory
   mkdir -p weights
   
   # Add your .pt model files to the weights directory
   cp /path/to/your/model.pt weights/
   ```

6. **Start the server**
   ```bash
   # Production mode
   uv run python main.py
   
   # Development mode
   uv run python main.py --dev
   ```

## 📁 Project Structure

```
backend/
├── api/                    # API routes and endpoints
│   └── routes.py          # FastAPI route definitions
├── models/                # Data models and configuration
│   ├── config.py          # Application configuration
│   └── detection.py       # Detection result models
├── services/              # Business logic services
│   ├── detection_service.py  # Main detection service
│   ├── model_service.py      # YOLO model management
│   └── image_service.py      # Image/video processing
├── static/                # Static files and processed content
│   ├── frames/            # Processed video frames
│   └── temp_frames/       # Temporary frame storage
├── weights/               # YOLO model weights (.pt files)
├── main.py               # Application entry point
├── app.py                # FastAPI application setup
├── start.sh              # Unix startup script
├── start.bat             # Windows startup script
├── test_gpu.py           # GPU acceleration test
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Project configuration
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Server Configuration
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
```

### Model Weights

Place your YOLO model weights (`.pt` files) in the `weights/` directory:

```
weights/
├── original.pt
├── tuned.pt
├── tuned-v2.pt
└── best.pt
```

## 🌐 API Endpoints

### Health & Status
- `GET /api/health` - Check API health and model status
- `GET /api/device` - Get GPU/device information
- `GET /api/config` - Get current configuration
- `POST /api/config` - Update configuration

### Model Management
- `GET /api/weights` - List available model weights
- `POST /api/weights/switch` - Switch to different model

### Detection
- `POST /api/images/detect` - Detect logos in multiple images
- `POST /api/video/detect` - Detect logos in video (streaming)

### API Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## 🎯 Usage Examples

### Image Detection
```bash
curl -X POST "http://localhost:8000/api/images/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "confidence_threshold=0.5"
```

### Video Detection
```bash
curl -X POST "http://localhost:8000/api/video/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@video.mp4" \
  -F "frames_per_second=2" \
  -F "confidence_threshold=0.5"
```

### Switch Model
```bash
curl -X POST "http://localhost:8000/api/weights/switch" \
  -H "Content-Type: application/json" \
  -d '{"weight_name": "tuned-v2.pt"}'
```

## 🚀 GPU Acceleration

The API automatically detects and uses GPU acceleration:

- **Apple Silicon (M1/M2/M3)**: Uses MPS (Metal Performance Shaders)
- **NVIDIA GPUs**: Uses CUDA (if available)
- **Fallback**: Uses CPU if no GPU is available

### Testing GPU
```bash
uv run python test_gpu.py
```

Expected output:
```
🚀 Using MPS (Metal Performance Shaders) for Apple Silicon
✅ GPU acceleration test completed successfully!
```

## 🔍 Troubleshooting

### Common Issues

1. **"MPS not available"**
   - Ensure you're on macOS 12.3+
   - Update PyTorch to 2.1.0+
   - Check Apple Silicon Mac compatibility

2. **"FFmpeg not found"**
   - Install FFmpeg: `brew install ffmpeg` (macOS)
   - Video processing won't work without FFmpeg

3. **"No model weights found"**
   - Add `.pt` files to the `weights/` directory
   - Check file permissions

4. **"Port already in use"**
   - Change port: `./start.sh --port 8080`
   - Kill existing process: `lsof -ti:8000 | xargs kill`

### Performance Optimization

1. **GPU Memory Issues**
   - Reduce batch size
   - Clear GPU cache periodically
   - Use smaller models

2. **Slow Inference**
   - Enable GPU acceleration
   - Use optimized model weights
   - Adjust confidence threshold

3. **Video Processing**
   - Reduce frames per second
   - Use shorter videos for testing
   - Ensure FFmpeg is installed

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
uv sync --extra dev

# Run GPU test
uv run python test_gpu.py

# Run API tests (if available)
uv run pytest
```

### Manual Testing
```bash
# Start server
./start.sh --dev

# Test endpoints
curl http://localhost:8000/api/health
curl http://localhost:8000/api/device
```

## 📊 Monitoring

### Logs
- Application logs are printed to console
- Look for "🚀 Using MPS" for GPU confirmation
- Monitor inference times in detection logs

### Performance Metrics
- GPU memory usage (MPS doesn't expose detailed metrics)
- Inference time per frame
- API response times

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Run `./start.sh --help` for script options
3. Check API documentation at `http://localhost:8000/docs`
4. Review logs for error messages
# brand-detector
