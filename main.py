import uvicorn
import argparse
import os
from app import app

def main():
    """Production server startup"""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Logo Detection API on {host}:{port}")
    print("📝 Production mode - optimized for performance")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

def dev():
    """Development server startup with auto-reload"""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Logo Detection API (DEV) on {host}:{port}")
    print("🔧 Development mode - auto-reload enabled")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True,
        reload_dirs=["."],
        log_level="debug",
        access_log=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logo Detection API Server")
    parser.add_argument("--dev", action="store_true", help="Run in development mode with auto-reload")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    
    args = parser.parse_args()
    
    # Set environment variables from command line args
    os.environ["PORT"] = str(args.port)
    os.environ["HOST"] = args.host
    
    if args.dev:
        dev()
    else:
        main()
