from typing import List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2

from models.config import ConfigUpdate
from services.detection_service import DetectionService
from services.image_service import ImageService


class WeightSwitchRequest(BaseModel):
    weight_name: str


class DetectionRoutes:
    def __init__(self, detection_service: DetectionService, image_service: ImageService):
        self.detection_service = detection_service
        self.image_service = image_service
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all routes"""
        
        @self.router.get("/")
        async def root():
            return {"message": "Logo Detection API", "status": "running"}
        
        @self.router.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_loaded": self.detection_service.is_model_loaded()
            }
        
        @self.router.get("/device")
        async def get_device_info():
            """Get information about the current device (GPU/CPU)"""
            return self.detection_service.model_service.get_device_info()
        
        @self.router.get("/config")
        async def get_config():
            return self.detection_service.config.to_dict()
        
        @self.router.post("/config")
        async def update_config(config_update: ConfigUpdate):
            self.detection_service.config.update(config_update)
            print(f"Configuration updated: {config_update.frames_per_second} frames per second, {config_update.confidence_threshold} confidence threshold")
            return {"message": "Configuration updated successfully"}
        
        @self.router.get("/weights")
        async def get_weights():
            """Get list of available weights"""
            return {
                "available_weights": self.detection_service.get_available_weights(),
                "current_weight": self.detection_service.get_current_weight()
            }
        
        @self.router.post("/weights/switch")
        async def switch_weight(request: WeightSwitchRequest):
            """Switch to a different weight"""
            success = self.detection_service.switch_weight(request.weight_name)
            if success:
                return {"message": f"Switched to weight: {request.weight_name}"}
            else:
                raise HTTPException(status_code=400, detail=f"Failed to switch to weight: {request.weight_name}")
        
        @self.router.post("/images/detect")
        async def detect_logos_images(
            files: List[UploadFile] = File(...),
            confidence_threshold: float = Form(0.5)
        ):
            if not self.detection_service.is_model_loaded():
                raise HTTPException(status_code=500, detail="Model not loaded")
            
            if confidence_threshold < 0.0 or confidence_threshold > 1.0:
                raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")
            
            try:
                results = []
                for file in files:
                        if not self.image_service.validate_image_file(file.content_type, file.filename):
                            results.append({
                                "detections": [],
                                "total_detections": 0,
                                "error": f"File {file.filename} is not a valid image"
                            })
                            continue
                        
                        contents = await file.read()
                        detections, annotated_image = self.detection_service.detect_in_image(
                            contents, confidence_threshold
                        )
                        
                        # Convert annotated image to base64 if available
                        annotated_image_b64 = None
                        if annotated_image is not None:
                            import base64
                            _, buffer = cv2.imencode('.jpg', annotated_image)
                            annotated_image_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
                        
                        results.append({
                            "detections": [detection.dict() for detection in detections],
                            "total_detections": len(detections),
                            "annotated_image": annotated_image_b64
                        })
                    
                return {"results": results}
                
            except Exception as e:
                print(f"Error processing images: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
        
        @self.router.post("/video/detect")
        async def detect_logos_video(
            file: UploadFile = File(...),
            frames_per_second: int = Form(2),
            confidence_threshold: float = Form(0.5)
        ):
            if not self.detection_service.is_model_loaded():
                raise HTTPException(status_code=500, detail="Model not loaded")
            
            if not self.image_service.validate_video_file(file.content_type, file.filename):
                raise HTTPException(status_code=400, detail="File must be a video")
            
            if frames_per_second < 1 or frames_per_second > 30:
                raise HTTPException(status_code=400, detail="Frames per second must be between 1 and 30")
            
            if confidence_threshold < 0.0 or confidence_threshold > 1.0:
                raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")
            
            try:
                contents = await file.read()
                return await self.detection_service.detect_video(
                    contents, file.filename, frames_per_second, confidence_threshold
                )
            
            except Exception as e:
                print(f"Error processing video {file.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    def get_router(self):
        """Get the configured router"""
        return self.router 