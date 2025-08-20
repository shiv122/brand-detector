import cv2
import numpy as np
import json
import asyncio
import os
import tempfile
import time
import subprocess
from pathlib import Path
from typing import AsyncGenerator, List, Tuple, Optional
from ultralytics import YOLO
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from models.config import AppConfig
from models.detection import Detection
from services.model_service import ModelService
from services.image_service import ImageService


class DetectionService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model_service = ModelService(config)
        self.image_service = ImageService()
        
        # Ensure directories exist
        self._setup_directories()
    
    def _setup_directories(self):
        """Setup required directories"""
        static_dir = Path(self.config.static_dir)
        frames_dir = Path(self.config.frames_dir)
        
        static_dir.mkdir(exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_service.is_loaded()
    
    def get_available_weights(self) -> List[dict]:
        """Get list of available weights"""
        return self.model_service.get_available_weights()
    
    def get_current_weight(self) -> str:
        """Get the currently selected weight"""
        return self.model_service.get_current_model_name()
    
    def switch_weight(self, weight_name: str) -> bool:
        """Switch to a different weight"""
        return self.model_service.switch_model(weight_name)
    
    def detect_in_image(self, image_data: bytes, confidence_threshold: float = 0.5) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """Detect logos in a single image"""
        return self.model_service.detect_in_image(image_data, confidence_threshold)
    
    async def detect_video(self, file_content: bytes, filename: str, frames_per_second: int, confidence_threshold: float) -> StreamingResponse:
        """Detect logos in video and stream results"""
        try:
            # Save video to permanent location
            video_filename = f"uploaded_{int(time.time())}_{filename}"
            video_path = Path(self.config.static_dir) / video_filename
            
            with open(video_path, 'wb') as f:
                f.write(file_content)
            
            # Get video information
            video_fps, total_frames, width, height = self.image_service.get_video_info(str(video_path))
            skip_frames = self.image_service.calculate_skip_frames(video_fps, frames_per_second)
            
            # Create processed video path
            processed_video_filename = f"processed_{int(time.time())}_{filename}"
            processed_video_path = Path(self.config.static_dir) / processed_video_filename
            
            # Create streaming response
            return StreamingResponse(
                self._generate_video_frames(str(video_path), str(processed_video_path), skip_frames, confidence_threshold),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        except Exception as e:
            # Clean up video file
            if 'video_path' in locals():
                try:
                    os.unlink(video_path)
                except:
                    pass
            
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    async def _generate_video_frames(self, video_path: str, processed_video_path: str, skip_frames: int, confidence_threshold: float) -> AsyncGenerator[str, None]:
        """Generate video frames with detections and create processed video using FFmpeg"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        processed_count = 0
        
        # Get video properties for output
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create temporary directory for frames
        temp_frames_dir = Path(self.config.static_dir) / "temp_frames"
        temp_frames_dir.mkdir(exist_ok=True)
        
        # Calculate estimated total processed frames
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        estimated_processed_frames = total_video_frames // skip_frames if skip_frames > 0 else total_video_frames
        
        # Store detection results for interpolation
        detection_results = {}  # frame_number -> (detections, annotated_frame)
        
        try:
            # Send initial status with estimated total frames
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting video processing...', 'estimated_total_frames': estimated_processed_frames})}\n\n"
            
            # First pass: Process frames at specified interval and store results
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frames at specified interval
                if frame_count % skip_frames == 0:
                    try:
                        # Run detection on frame
                        detections, annotated_frame = self.model_service.detect_in_frame(
                            frame, confidence_threshold
                        )
                        
                        if annotated_frame is not None:
                            # Store detection results for this frame
                            detection_results[frame_count] = (detections, annotated_frame)
                            
                            # Save frame to static directory for frontend display
                            frame_filename = f"frame_{processed_count:06d}.jpg"
                            frame_path = Path(self.config.frames_dir) / frame_filename
                            cv2.imwrite(str(frame_path), annotated_frame)
                            frame_path.touch()
                            await asyncio.sleep(0.01)
                            
                            frame_url = f"/static/frames/{frame_filename}"
                            
                            # Create frame data
                            frame_data = {
                                "frame_number": processed_count,
                                "frame_url": frame_url,
                                "detections": [detection.dict() for detection in detections],
                                "total_detections": len(detections),
                                "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS)
                            }
                            
                            # Send frame data
                            yield f"data: {json.dumps({'type': 'frame', **frame_data})}\n\n"
                            processed_count += 1
                    
                    except Exception as e:
                        print(f"Error processing frame {processed_count}: {str(e)}")
                        # Store original frame if detection failed
                        detection_results[frame_count] = ([], frame)
                
                frame_count += 1
            
            # Send completion message immediately after detection phase
            processed_video_url = f"/static/{Path(processed_video_path).name}"
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Video processing completed', 'total_frames': processed_count, 'processed_video_url': processed_video_url})}\n\n"
            
            # Reset video capture for second pass
            cap.release()
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            # Second pass: Create consistent video with interpolated detections
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Find the nearest processed frame for interpolation
                nearest_frame = self._find_nearest_processed_frame(frame_count, detection_results.keys())
                
                if nearest_frame in detection_results:
                    detections, annotated_frame = detection_results[nearest_frame]
                    
                    # Apply the same detections to current frame
                    if detections:
                        # Create annotated frame with same detections
                        annotated_frame = self._apply_detections_to_frame(frame, detections)
                    else:
                        annotated_frame = frame
                else:
                    # No nearby detections, use original frame
                    annotated_frame = frame
                
                # Save frame to temp directory for video creation
                frame_filename = f"frame_{frame_count:06d}.jpg"
                temp_frame_path = temp_frames_dir / frame_filename
                cv2.imwrite(str(temp_frame_path), annotated_frame)
                temp_frame_path.touch()
                
                frame_count += 1
            
            # Use FFmpeg to create processed video from frames
            await self._create_video_from_frames(temp_frames_dir, processed_video_path, fps, frame_count)
            
            # Clean up temp frames directory
            import shutil
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
            # Send video creation completion message
            yield f"data: {json.dumps({'type': 'video_ready', 'message': 'Video with detections created successfully', 'processed_video_url': processed_video_url})}\n\n"
        
        finally:
            cap.release()
            # Clean up original video file
            try:
                os.unlink(video_path)
            except:
                pass
    
    async def _create_video_from_frames(self, frames_dir: Path, output_path: str, fps: int, total_frames: int) -> None:
        """Create MP4 video from frames using FFmpeg"""
        try:
            # FFmpeg command to create MP4 video from frames
            # -y: overwrite output file
            # -framerate: set input frame rate
            # -i: input pattern for frames
            # -c:v libx264: use H.264 codec
            # -preset fast: encoding preset for speed
            # -crf 23: constant rate factor for quality
            # -pix_fmt yuv420p: pixel format for compatibility
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-framerate', str(fps),
                '-i', str(frames_dir / 'frame_%06d.jpg'),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',  # Optimize for web streaming
                output_path
            ]
            
            # Run FFmpeg command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                print(f"FFmpeg error: {stderr.decode()}")
                raise Exception(f"FFmpeg failed to create video: {stderr.decode()}")
            
            print(f"Successfully created processed video: {output_path}")
            
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Please install FFmpeg to process videos.")
        except Exception as e:
            raise Exception(f"Error creating video with FFmpeg: {str(e)}")
    
    def _find_nearest_processed_frame(self, current_frame: int, processed_frames: list) -> int:
        """Find the nearest processed frame to the current frame"""
        if not processed_frames:
            return current_frame
        
        processed_frames = sorted(processed_frames)
        
        # Find the closest processed frame
        nearest = processed_frames[0]
        min_distance = abs(current_frame - nearest)
        
        for frame in processed_frames:
            distance = abs(current_frame - frame)
            if distance < min_distance:
                min_distance = distance
                nearest = frame
        
        return nearest
    
    def _apply_detections_to_frame(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Apply detections to a frame by drawing bounding boxes"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = detection.bbox
            
            # Convert to integers for drawing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{detection.class_name} {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    # Removed get_frames_status method # Removed frame caching
    # def get_frames_status(self) -> Dict[str, Any]: # Removed frame caching
    #     """Get status of processed frames""" # Removed frame caching
    #     frames_dir = Path(self.config.frames_dir) # Removed frame caching
    #     saved_frames = [] # Removed frame caching
        
    #     if frames_dir.exists(): # Removed frame caching
    #         saved_frames = [f.name for f in frames_dir.glob("frame_*.jpg")] # Removed frame caching
        
    #     return { # Removed frame caching
    #         "total_frames": len(self.detection_cache.get_all_frames()), # Removed frame caching
    #         "frame_numbers": list(self.detection_cache.get_all_frames().keys()), # Removed frame caching
    #         "frames_with_detections": self.detection_cache.get_frames_with_detections(), # Removed frame caching
    #         "saved_frame_files": saved_frames, # Removed frame caching
    #         "frames_dir_exists": frames_dir.exists(), # Removed frame caching
    #         "frames_dir_path": str(frames_dir.absolute()) # Removed frame caching
    #     } 