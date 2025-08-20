import base64
import io
from typing import Optional
import cv2
import numpy as np
from PIL import Image
from models.detection import Detection


class ImageService:
    @staticmethod
    def image_to_base64(image_np: np.ndarray) -> str:
        """Convert numpy array image to base64 string"""
        try:
            # Convert BGR to RGB if needed
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_np
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
            return ""
    
    @staticmethod
    def save_frame(frame: np.ndarray, frame_path: str, quality: int = 85) -> bool:
        """Save a frame to disk"""
        try:
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Save using PIL for better quality control
            pil_image = Image.fromarray(frame_rgb)
            pil_image.save(frame_path, quality=quality, optimize=False)
            
            return True
        except Exception as e:
            print(f"Error saving frame {frame_path}: {str(e)}")
            return False
    
    @staticmethod
    def validate_image_file(content_type: str, filename: str) -> bool:
        """Validate if file is an image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Check MIME type
        if content_type.startswith("image/"):
            return True
        
        # Check file extension
        if filename:
            file_extension = '.' + filename.split('.')[-1].lower()
            return file_extension in image_extensions
        
        return False
    
    @staticmethod
    def validate_video_file(content_type: str, filename: str) -> bool:
        """Validate if file is a video"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        # Check MIME type
        if content_type.startswith("video/"):
            return True
        
        # Check file extension
        if filename:
            file_extension = '.' + filename.split('.')[-1].lower()
            return file_extension in video_extensions
        
        return False
    
    @staticmethod
    def get_video_info(video_path: str) -> tuple:
        """Get video information (fps, total frames)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return fps, total_frames, width, height
    
    @staticmethod
    def calculate_skip_frames(video_fps: int, target_fps: int) -> int:
        """Calculate how many frames to skip to achieve target FPS"""
        return max(1, video_fps // target_fps) 