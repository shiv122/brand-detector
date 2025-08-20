from typing import List, Optional
from pydantic import BaseModel


class Detection(BaseModel):
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str


class VideoFrameData(BaseModel):
    frame_number: int
    frame_url: str
    detections: List[Detection]
    total_detections: int
    timestamp: float