import os
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from models.detection import Detection
from models.config import AppConfig, WeightInfo


class ModelService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.models: dict = {}  # Cache for loaded models
        self.current_model: Optional[YOLO] = None
        self.device = self._get_optimal_device()
        self._load_available_weights()
        self._load_default_model()
    
    def _get_optimal_device(self) -> str:
        """Get the optimal device for inference"""
        if torch.backends.mps.is_available():
            print("🚀 Using MPS (Metal Performance Shaders) for Apple Silicon")
            return "mps"
        elif torch.cuda.is_available():
            print("🚀 Using CUDA GPU")
            return "cuda"
        else:
            print("⚠️ Using CPU (no GPU acceleration)")
            return "cpu"
    
    def _load_available_weights(self):
        """Load all available weights from the weights directory"""
        weights_dir = Path(self.config.weights_dir)
        if not weights_dir.exists():
            print(f"❌ Weights directory not found: {weights_dir}")
            return
        
        for weight_file in weights_dir.glob("*.pt"):
            try:
                size = weight_file.stat().st_size
                self.config.available_weights.append(WeightInfo(
                    name=weight_file.name,
                    path=str(weight_file),
                    size=size,
                    description=f"YOLO model ({self._format_size(size)})"
                ))
                print(f"✅ Found weight: {weight_file.name} ({self._format_size(size)})")
            except Exception as e:
                print(f"❌ Error loading weight {weight_file.name}: {str(e)}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def _load_default_model(self):
        """Load the default model"""
        self.switch_model(self.config.selected_weight)
    
    def switch_model(self, weight_name: str) -> bool:
        """Switch to a different weight file"""
        try:
            print(f"🔄 Attempting to switch to model: {weight_name}")
            
            # Check if weight exists
            weight_path = Path(self.config.weights_dir) / weight_name
            if not weight_path.exists():
                print(f"❌ Weight file not found: {weight_path}")
                return False
            
            # Update config
            self.config.set_selected_weight(weight_name)
            print(f"📝 Updated config selected_weight to: {weight_name}")
            
            # Load model if not already cached
            if weight_name not in self.models:
                print(f"🔄 Loading model: {weight_name} on {self.device}")
                model = YOLO(str(weight_path))
                # Move model to optimal device with optimizations
                model.to(self.device)
                
                # Enable optimizations for MPS
                if self.device == "mps":
                    # Set model to evaluation mode for inference
                    model.model.eval()
                    # Enable memory efficient attention if available
                    if hasattr(model.model, 'enable_memory_efficient_attention'):
                        model.model.enable_memory_efficient_attention()
                
                self.models[weight_name] = model
                print(f"✅ Model loaded successfully: {weight_name} on {self.device}")
            else:
                print(f"📦 Using cached model: {weight_name}")
            
            # Set as current model
            self.current_model = self.models[weight_name]
            print(f"✅ Switched to model: {weight_name}")
            print(f"🔍 Current model config: {self.config.selected_weight}")
            return True
            
        except Exception as e:
            print(f"❌ Error switching to model {weight_name}: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if any model is loaded"""
        return self.current_model is not None
    
    def get_current_model_name(self) -> str:
        """Get the name of the currently loaded model"""
        return self.config.selected_weight
    
    def get_available_weights(self) -> List[dict]:
        """Get list of available weights"""
        return [weight.dict() for weight in self.config.available_weights]
    
    def get_device_info(self) -> dict:
        """Get information about the current device"""
        info = {
            "device": self.device,
            "device_name": "Unknown"
        }
        
        if self.device == "mps":
            info["device_name"] = "Apple Silicon MPS"
            # Get MPS memory info if available
            try:
                if hasattr(torch.mps, 'get_device_properties'):
                    props = torch.mps.get_device_properties()
                    info["memory_total"] = props.total_memory if hasattr(props, 'total_memory') else "Unknown"
            except:
                pass
        elif self.device == "cuda":
            info["device_name"] = torch.cuda.get_device_name(0)
            info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["memory_allocated"] = torch.cuda.memory_allocated(0)
            info["memory_cached"] = torch.cuda.memory_reserved(0)
        else:
            info["device_name"] = "CPU"
        
        return info
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.device == "mps":
            # MPS doesn't have empty_cache, but we can try to free memory
            try:
                import gc
                gc.collect()
            except:
                pass
        elif self.device == "cuda":
            torch.cuda.empty_cache()
    
    def detect_in_image(self, image_data: bytes, confidence_threshold: float = 0.5) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """Detect logos in a single image"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Run detection with device optimization and batch processing
        results = self.current_model(
            img, 
            save=False, 
            conf=confidence_threshold, 
            device=self.device,
            verbose=False  # Reduce logging for faster inference
        )
        
        detections = []
        annotated_img = None
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Use device-appropriate tensor operations
                    if self.device == "mps":
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                    else:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                    class_name = self.current_model.names[class_id]
                    
                    detections.append(Detection(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    ))
            
            # Get annotated image
            annotated_img = result.plot()
        
        return detections, annotated_img
    
    def detect_in_frame(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """Detect logos in a video frame"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Debug: Print which model is being used
        print(f"🔍 Using model: {self.config.selected_weight} for detection")
        
        # Run detection with device optimization and batch processing
        results = self.current_model(
            frame, 
            save=False, 
            conf=confidence_threshold, 
            device=self.device,
            verbose=False  # Reduce logging for faster inference
        )
        
        detections = []
        annotated_frame = None
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Use device-appropriate tensor operations
                    if self.device == "mps":
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                    else:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                    
                    class_name = self.current_model.names[class_id]
                    
                    detections.append(Detection(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    ))
            
            # Get annotated frame
            annotated_frame = result.plot()
        
        return detections, annotated_frame 