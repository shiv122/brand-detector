from typing import List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from services.classification_service import ClassificationService
from services.image_service import ImageService


class WeightSwitchRequest(BaseModel):
    weight_name: str


class ClassificationRoutes:
    def __init__(
        self, classification_service: ClassificationService, image_service: ImageService
    ):
        self.classification_service = classification_service
        self.image_service = image_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Setup all classification routes"""

        @self.router.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_loaded": self.classification_service.is_model_loaded(),
            }

        @self.router.get("/weights")
        async def get_weights():
            """Get list of available classification weights"""
            return {
                "available_weights": self.classification_service.get_available_weights(),
                "current_weight": self.classification_service.get_current_weight(),
            }

        @self.router.post("/weights/switch")
        async def switch_weight(request: WeightSwitchRequest):
            """Switch to a different classification weight"""
            success = self.classification_service.switch_weight(request.weight_name)
            if success:
                return {"message": f"Switched to weight: {request.weight_name}"}
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to switch to weight: {request.weight_name}",
                )

        @self.router.post("/images/classify")
        async def classify_images(
            files: List[UploadFile] = File(...),
            top_k: int = Form(5)
        ):
            """Classify images and return top-k predictions"""
            if not self.classification_service.is_model_loaded():
                raise HTTPException(status_code=500, detail="Classification model not loaded")

            if top_k < 1 or top_k > 20:
                raise HTTPException(
                    status_code=400,
                    detail="top_k must be between 1 and 20",
                )

            try:
                results = []
                for file in files:
                    if not self.image_service.validate_image_file(
                        file.content_type, file.filename
                    ):
                        results.append(
                            {
                                "classifications": [],
                                "error": f"File {file.filename} is not a valid image",
                            }
                        )
                        continue

                    contents = await file.read()
                    classifications = self.classification_service.classify_image(
                        contents, top_k
                    )

                    results.append(
                        {
                            "classifications": [
                                classification.dict() for classification in classifications
                            ],
                            "top_prediction": classifications[0].dict() if classifications else None,
                        }
                    )

                return {"results": results}

            except Exception as e:
                print(f"Error processing images: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Error processing images: {str(e)}"
                )

    def get_router(self):
        """Get the configured router"""
        return self.router


