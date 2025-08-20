from pydantic import BaseModel
from typing import Optional, List


class ConfigUpdate(BaseModel):
    frames_per_second: int
    confidence_threshold: float


class WeightInfo(BaseModel):
    name: str
    path: str
    size: int
    description: Optional[str] = None


class AppConfig:
    def __init__(self):
        self.frames_per_second: int = 2
        self.confidence_threshold: float = 0.5
        self.weights_dir: str = "weights"
        self.static_dir: str = "static"
        self.frames_dir: str = "static/frames"
        self.available_weights: List[WeightInfo] = []
        self.selected_weight: str = "original.pt"  
    
    def update(self, config_update: ConfigUpdate):
        self.frames_per_second = config_update.frames_per_second
        self.confidence_threshold = config_update.confidence_threshold
    
    def set_selected_weight(self, weight_name: str):
        """Set the selected weight for detection"""
        self.selected_weight = weight_name
    
    def get_weight_path(self) -> str:
        """Get the full path to the selected weight"""
        return f"{self.weights_dir}/{self.selected_weight}"
    
    def to_dict(self):
        return {
            "frames_per_second": self.frames_per_second,
            "confidence_threshold": self.confidence_threshold,
            "selected_weight": self.selected_weight,
            "available_weights": [weight.dict() for weight in self.available_weights]
        } 