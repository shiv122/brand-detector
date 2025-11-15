import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from models.config import AppConfig
from models.detection import ClassificationResult
from services.classification_model_service import ClassificationModelService
from services.image_service import ImageService


class ClassificationService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model_service = ClassificationModelService(config)
        self.image_service = ImageService()
        
        # Load default model if available
        self._load_default_model()
    
    def _load_default_model(self):
        """Load the first available classification model"""
        available_weights = self.model_service.get_available_weights()
        if available_weights and len(available_weights) > 0:
            first_weight = available_weights[0]["name"]
            success = self.model_service.switch_model(first_weight)
            if success:
                print(f"✅ Loaded default classification model: {first_weight}")
            else:
                print(f"⚠️ Failed to load default classification model: {first_weight}")
        else:
            print("⚠️ No classification weights found in weights/classification_weights directory")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_service.is_loaded()
    
    def get_available_weights(self) -> List[dict]:
        """Get list of available classification weights"""
        return self.model_service.get_available_weights()
    
    def get_current_weight(self) -> str:
        """Get the currently selected weight"""
        return self.model_service.get_current_model_name()
    
    def switch_weight(self, weight_name: str) -> bool:
        """Switch to a different weight"""
        return self.model_service.switch_model(weight_name)
    
    def classify_image(
        self, image_data: bytes, top_k: int = 5
    ) -> List[ClassificationResult]:
        """Classify an image and return top-k predictions"""
        if not self.is_model_loaded():
            raise RuntimeError("Classification model not loaded")
        
        results = self.model_service.classify_image(image_data, top_k)
        
        # Convert to ClassificationResult objects
        classification_results = [
            ClassificationResult(
                class_id=r["class_id"],
                class_name=r["class_name"],
                confidence=r["confidence"]
            )
            for r in results
        ]
        
        return classification_results

