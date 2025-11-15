import os
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from models.config import AppConfig, WeightInfo


class ClassificationModelService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.models: dict = {}  # Cache for loaded models
        self.current_model: Optional[YOLO] = None
        self.device = self._get_optimal_device()
        self.classification_weights_dir = Path(self.config.weights_dir) / "classification_weights"
        self.classification_weights: List[WeightInfo] = []
        self._load_available_weights()
    
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
        """Load all available weights from the classification_weights directory"""
        if not self.classification_weights_dir.exists():
            print(f"❌ Classification weights directory not found: {self.classification_weights_dir}")
            return
        
        # Store classification weights separately
        self.classification_weights: List[WeightInfo] = []
        
        for weight_file in self.classification_weights_dir.glob("*.pt"):
            try:
                size = weight_file.stat().st_size
                weight_info = WeightInfo(
                    name=weight_file.name,
                    path=str(weight_file),
                    size=size,
                    description=f"YOLO Classification model ({self._format_size(size)})"
                )
                self.classification_weights.append(weight_info)
                print(f"✅ Found classification weight: {weight_file.name} ({self._format_size(size)})")
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
    
    def switch_model(self, weight_name: str) -> bool:
        """Switch to a different weight file"""
        try:
            print(f"🔄 Attempting to switch to classification model: {weight_name}")
            
            # Check if weight exists in classification_weights directory
            weight_path = self.classification_weights_dir / weight_name
            if not weight_path.exists():
                print(f"❌ Classification weight file not found: {weight_path}")
                return False
            
            # Load model if not already cached
            if weight_name not in self.models:
                print(f"🔄 Loading classification model: {weight_name} on {self.device}")
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
                print(f"✅ Classification model loaded successfully: {weight_name} on {self.device}")
            else:
                print(f"📦 Using cached classification model: {weight_name}")
            
            # Set as current model
            self.current_model = self.models[weight_name]
            print(f"✅ Switched to classification model: {weight_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error switching to classification model {weight_name}: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if any model is loaded"""
        return self.current_model is not None
    
    def get_current_model_name(self) -> str:
        """Get the name of the currently loaded model"""
        if self.current_model is None:
            return ""
        # Extract name from model path
        for name, model in self.models.items():
            if model == self.current_model:
                return name
        return ""
    
    def get_available_weights(self) -> List[dict]:
        """Get list of available classification weights"""
        return [weight.dict() for weight in self.classification_weights]
    
    def classify_image(self, image_data: bytes, top_k: int = 5) -> List[dict]:
        """Classify an image and return top-k predictions"""
        if not self.is_loaded():
            raise RuntimeError("Classification model not loaded")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Run classification
        results = self.current_model(
            img,
            save=False,
            device=self.device,
            verbose=False
        )
        
        classifications = []
        
        for result in results:
            # For classification models, results have probs attribute
            if hasattr(result, 'probs'):
                probs = result.probs
                
                # Get all probabilities
                if hasattr(probs, 'data'):
                    # Get probabilities tensor
                    probs_tensor = probs.data
                    if self.device == "mps" or self.device == "cuda":
                        all_probs = probs_tensor.cpu().numpy()
                    else:
                        all_probs = probs_tensor.numpy()
                    
                    # Get top-k indices and confidences
                    top_k_indices = np.argsort(all_probs)[-top_k:][::-1]
                    top_k_confidences = all_probs[top_k_indices]
                elif hasattr(probs, 'top5') and hasattr(probs, 'top5conf'):
                    # Use built-in top5 if available
                    top5_indices = probs.top5[:top_k]
                    top5_confidences = probs.top5conf[:top_k]
                    top_k_indices = top5_indices
                    top_k_confidences = top5_confidences
                else:
                    # Fallback: try to get probabilities directly
                    try:
                        all_probs = np.array(probs)
                        top_k_indices = np.argsort(all_probs)[-top_k:][::-1]
                        top_k_confidences = all_probs[top_k_indices]
                    except:
                        print("Warning: Could not extract probabilities from classification result")
                        continue
                
                # Get class names from model or result
                if hasattr(result, 'names') and result.names:
                    class_names = result.names
                elif hasattr(self.current_model, 'names') and self.current_model.names:
                    class_names = self.current_model.names
                else:
                    class_names = {}
                
                for idx, conf in zip(top_k_indices, top_k_confidences):
                    class_id = int(idx)
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    confidence = float(conf)
                    
                    classifications.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence
                    })
            else:
                print("Warning: Classification result does not have probs attribute")
        
        # Sort by confidence descending (should already be sorted, but ensure it)
        classifications.sort(key=lambda x: x["confidence"], reverse=True)
        
        return classifications[:top_k]

