from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from models.config import AppConfig
from services.detection_service import DetectionService
from services.classification_service import ClassificationService
from services.image_service import ImageService
from api.routes import DetectionRoutes
from api.classification_routes import ClassificationRoutes


class LogoDetectionApp:
    def __init__(self):
        self.config = AppConfig()
        self.detection_service = DetectionService(self.config)
        self.classification_service = ClassificationService(self.config)
        self.image_service = ImageService()
        self.detection_routes = DetectionRoutes(self.detection_service, self.image_service)
        self.classification_routes = ClassificationRoutes(self.classification_service, self.image_service)
        self.app = None
    
    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            print("🚀 Starting Logo Detection API...")
            if self.detection_service.is_model_loaded():
                print("✅ Detection model loaded successfully")
            else:
                print("❌ Failed to load detection model")
            
            if self.classification_service.is_model_loaded():
                print("✅ Classification model loaded successfully")
            else:
                print("⚠️ No classification model loaded")
            
            yield
            
            # Shutdown
            print("🛑 Shutting down Logo Detection API...")
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Logo Detection API",
            description="API for detecting logos in images and videos using YOLO",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files
        static_dir = Path(self.config.static_dir)
        static_dir.mkdir(exist_ok=True)
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # Include routes
        self.app.include_router(self.detection_routes.get_router(), prefix="/api")
        self.app.include_router(self.classification_routes.get_router(), prefix="/api/classification")
        
        return self.app
    
    def get_app(self) -> FastAPI:
        """Get the configured FastAPI application"""
        if self.app is None:
            self.app = self.create_app()
        return self.app


# Create app instance
app_instance = LogoDetectionApp()
app = app_instance.get_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 