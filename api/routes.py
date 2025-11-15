from typing import List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import cv2

from models.config import ConfigUpdate
from services.detection_service import DetectionService
from services.image_service import ImageService


class WeightSwitchRequest(BaseModel):
    weight_name: str


class CSVExportRequest(BaseModel):
    session_id: str
    filename_prefix: str = None


class DetectionRoutes:
    def __init__(
        self, detection_service: DetectionService, image_service: ImageService
    ):
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
                "model_loaded": self.detection_service.is_model_loaded(),
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
            print(
                f"Configuration updated: {config_update.frames_per_second} frames per second, {config_update.confidence_threshold} confidence threshold"
            )
            return {"message": "Configuration updated successfully"}

        @self.router.get("/weights")
        async def get_weights():
            """Get list of available weights"""
            return {
                "available_weights": self.detection_service.get_available_weights(),
                "current_weight": self.detection_service.get_current_weight(),
            }

        @self.router.post("/weights/switch")
        async def switch_weight(request: WeightSwitchRequest):
            """Switch to a different weight"""
            success = self.detection_service.switch_weight(request.weight_name)
            if success:
                return {"message": f"Switched to weight: {request.weight_name}"}
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to switch to weight: {request.weight_name}",
                )

        @self.router.post("/images/detect")
        async def detect_logos_images(
            files: List[UploadFile] = File(...), confidence_threshold: float = Form(0.5)
        ):
            if not self.detection_service.is_model_loaded():
                raise HTTPException(status_code=500, detail="Model not loaded")

            if confidence_threshold < 0.0 or confidence_threshold > 1.0:
                raise HTTPException(
                    status_code=400,
                    detail="Confidence threshold must be between 0.0 and 1.0",
                )

            try:
                results = []
                for file in files:
                    if not self.image_service.validate_image_file(
                        file.content_type, file.filename
                    ):
                        results.append(
                            {
                                "detections": [],
                                "total_detections": 0,
                                "error": f"File {file.filename} is not a valid image",
                            }
                        )
                        continue

                    contents = await file.read()
                    detections, annotated_image = (
                        self.detection_service.detect_in_image(
                            contents, confidence_threshold
                        )
                    )

                    # Convert annotated image to base64 if available
                    annotated_image_b64 = None
                    if annotated_image is not None:
                        import base64

                        _, buffer = cv2.imencode(".jpg", annotated_image)
                        annotated_image_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

                    results.append(
                        {
                            "detections": [
                                detection.dict() for detection in detections
                            ],
                            "total_detections": len(detections),
                            "annotated_image": annotated_image_b64,
                        }
                    )

                return {"results": results}

            except Exception as e:
                print(f"Error processing images: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Error processing images: {str(e)}"
                )

        @self.router.post("/video/detect")
        async def detect_logos_video(
            file: UploadFile = File(None),
            file_url: str = Form(None),
            frames_per_second: int = Form(2),
            confidence_threshold: float = Form(0.5),
        ):
            if not self.detection_service.is_model_loaded():
                raise HTTPException(status_code=500, detail="Model not loaded")

            if frames_per_second < 1 or frames_per_second > 30:
                raise HTTPException(
                    status_code=400, detail="Frames per second must be between 1 and 30"
                )

            if confidence_threshold < 0.0 or confidence_threshold > 1.0:
                raise HTTPException(
                    status_code=400,
                    detail="Confidence threshold must be between 0.0 and 1.0",
                )

            # Validate that either file or file_url is provided
            if not file and not file_url:
                raise HTTPException(
                    status_code=400, detail="Either file or file_url must be provided"
                )

            if file and file_url:
                raise HTTPException(
                    status_code=400, detail="Provide either file or file_url, not both"
                )

            try:
                if file_url:
                    # Download video from URL
                    return await self.detection_service.detect_video_from_url(
                        file_url, frames_per_second, confidence_threshold
                    )
                else:
                    # Process uploaded file
                    if not self.image_service.validate_video_file(
                        file.content_type, file.filename
                    ):
                        raise HTTPException(status_code=400, detail="File must be a video")

                    contents = await file.read()
                    return await self.detection_service.detect_video(
                        contents, file.filename, frames_per_second, confidence_threshold
                    )

            except Exception as e:
                error_msg = f"Error processing video: {str(e)}"
                if file:
                    error_msg = f"Error processing video {file.filename}: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

        @self.router.get("/session/{session_id}/summary")
        async def get_session_summary(session_id: str):
            """Get summary of detection session"""
            try:
                summary = self.detection_service.get_session_summary(session_id)
                return summary
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error getting session summary: {str(e)}"
                )

        @self.router.get("/session/{session_id}/realtime-csv")
        async def get_realtime_csv_files(session_id: str):
            """Get real-time CSV files for a session"""
            try:
                csv_files = self.detection_service.get_realtime_csv_files(session_id)
                return {"csv_files": csv_files, "session_id": session_id}
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error getting real-time CSV files: {str(e)}",
                )

        @self.router.post("/session/export-csv")
        async def export_session_to_csv(request: CSVExportRequest):
            """Export session data to CSV files"""
            try:
                csv_files = self.detection_service.export_session_to_csv(
                    request.session_id, request.filename_prefix
                )
                return {
                    "message": "CSV files exported successfully",
                    "csv_files": csv_files,
                    "session_id": request.session_id,
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error exporting CSV: {str(e)}"
                )

        @self.router.get("/csv-files")
        async def get_available_csv_files():
            """Get list of available CSV files"""
            try:
                csv_files = self.detection_service.get_available_csv_files()
                return {"csv_files": csv_files}
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error getting CSV files: {str(e)}"
                )

        @self.router.get("/csv-files/download/{filename}")
        async def download_csv_file(filename: str):
            """Download a specific CSV file"""
            try:
                from pathlib import Path

                csv_dir = Path(self.detection_service.config.static_dir) / "csv_reports"
                file_path = csv_dir / filename

                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")

                return FileResponse(
                    path=str(file_path), filename=filename, media_type="text/csv"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error downloading file: {str(e)}"
                )

        @self.router.delete("/csv-files/cleanup")
        async def cleanup_old_csv_files(max_files: int = Query(50, ge=1, le=200)):
            """Clean up old CSV files"""
            try:
                self.detection_service.cleanup_old_csv_files(max_files)
                return {
                    "message": f"Cleaned up old CSV files, keeping {max_files} most recent"
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error cleaning up files: {str(e)}"
                )

    def get_router(self):
        """Get the configured router"""
        return self.router
