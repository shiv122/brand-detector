import cv2
import numpy as np
import json
import asyncio
import os
import tempfile
import time
import subprocess
import secrets
import httpx
from pathlib import Path
from typing import AsyncGenerator, List, Tuple, Optional
from ultralytics import YOLO
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from models.config import AppConfig
from models.detection import Detection, ClassificationResult
from services.model_service import ModelService
from services.image_service import ImageService
from services.counting_service import LogoCountingService
from services.classification_service import ClassificationService


class DetectionService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model_service = ModelService(config)
        self.image_service = ImageService()
        self.counting_service = LogoCountingService(config.static_dir)
        self.classification_service = ClassificationService(config)

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

    def _crop_detection_box(self, frame: np.ndarray, bbox: List[float], padding: int = 40) -> np.ndarray:
        """Crop detection box from frame with padding"""
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        
        # Add padding and ensure within bounds
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(width, int(x2) + padding)
        y2 = min(height, int(y2) + padding)
        
        # Crop the region
        cropped = frame[y1:y2, x1:x2]
        return cropped
    
    def _classify_detection(self, frame: np.ndarray, detection: Detection) -> Optional[List[ClassificationResult]]:
        """Classify a detection by cropping and running classification model"""
        if not self.classification_service.is_model_loaded():
            return None
        
        try:
            # Crop detection box with padding
            cropped = self._crop_detection_box(frame, detection.bbox, padding=40)
            
            if cropped.size == 0:
                return None
            
            # Convert to bytes for classification
            _, buffer = cv2.imencode('.jpg', cropped)
            image_bytes = buffer.tobytes()
            
            # Run classification
            classification_results = self.classification_service.classify_image(image_bytes, top_k=3)
            
            return classification_results
        except Exception as e:
            print(f"[CLASSIFICATION] Error classifying detection: {str(e)}")
            return None

    def detect_in_image(
        self, image_data: bytes, confidence_threshold: float = 0.5
    ) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """Detect logos in a single image"""
        return self.model_service.detect_in_image(image_data, confidence_threshold)

    async def detect_video_from_url(
        self,
        file_url: str,
        frames_per_second: int,
        confidence_threshold: float,
        create_video: bool = False,
        enable_classification: bool = False,
    ) -> StreamingResponse:
        """Detect logos in video from URL and stream results"""
        try:
            # Extract filename from URL or generate one
            filename = file_url.split("/")[-1].split("?")[0] or "video.mp4"

            # Validate URL
            if not file_url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail="Invalid URL format")

            # Create streaming response that includes download progress
            return StreamingResponse(
                self._download_and_process_video_from_url(
                    file_url,
                    filename,
                    frames_per_second,
                    confidence_threshold,
                    create_video,
                    enable_classification,
                ),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing video from URL: {str(e)}"
            )

    async def _download_and_process_video_from_url(
        self,
        file_url: str,
        filename: str,
        frames_per_second: int,
        confidence_threshold: float,
        create_video: bool = False,
        enable_classification: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Download video from URL with progress updates, then process it"""
        video_path = None
        try:
            # Send initial download status
            print(f"[DOWNLOAD] Connecting to server: {file_url}")
            print(f"[DOWNLOAD] Create video: {create_video}")
            yield f"data: {json.dumps({'type': 'download_status', 'status': 'Connecting to server...', 'percentage': 0})}\n\n"

            # Download video from URL
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("GET", file_url) as response:
                    if response.status_code != 200:
                        error_msg = (
                            f"Failed to download video from URL: {response.status_code}"
                        )
                        print(f"[DOWNLOAD ERROR] {error_msg}")
                        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                        return

                    # Save video to permanent location
                    video_filename = f"uploaded_{int(time.time())}_{filename}"
                    video_path = Path(self.config.static_dir) / video_filename

                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    print(f"[DOWNLOAD] Starting download: {video_filename}")
                    if total_size > 0:
                        print(
                            f"[DOWNLOAD] Total size: {total_size / (1024*1024):.2f} MB"
                        )
                    else:
                        print(f"[DOWNLOAD] Total size: Unknown (streaming)")

                    last_progress_update = 0
                    # Update progress every 1% or every 1MB, whichever is smaller
                    progress_update_interval = (
                        max(1024 * 1024, total_size // 100)
                        if total_size > 0
                        else 1024 * 1024  # 1MB intervals if size unknown
                    )

                    with open(video_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Send progress updates periodically
                            if (
                                downloaded - last_progress_update
                                >= progress_update_interval
                            ):
                                last_progress_update = downloaded

                                if total_size > 0:
                                    # Calculate actual percentage (0-100%)
                                    percentage = int((downloaded / total_size) * 100)
                                    mb_downloaded = downloaded / (1024 * 1024)
                                    mb_total = total_size / (1024 * 1024)
                                    status_msg = f"Downloading... {mb_downloaded:.2f}MB / {mb_total:.2f}MB ({percentage}%)"
                                    print(f"[DOWNLOAD] {status_msg}")
                                    yield f"data: {json.dumps({'type': 'download_status', 'status': status_msg, 'percentage': percentage})}\n\n"
                                else:
                                    # If we don't know the total size, show downloaded amount
                                    mb_downloaded = downloaded / (1024 * 1024)
                                    # Estimate percentage based on reasonable video sizes (assume max 500MB for progress)
                                    estimated_percentage = min(
                                        95, int((mb_downloaded / 500) * 100)
                                    )
                                    status_msg = f"Downloaded {mb_downloaded:.2f}MB..."
                                    print(
                                        f"[DOWNLOAD] {status_msg} (estimated {estimated_percentage}%)"
                                    )
                                    yield f"data: {json.dumps({'type': 'download_status', 'status': status_msg, 'percentage': estimated_percentage})}\n\n"

            # Download complete
            print(f"[DOWNLOAD] Download complete: {downloaded / (1024*1024):.2f} MB")
            yield f"data: {json.dumps({'type': 'download_status', 'status': 'Download complete, starting processing...', 'percentage': 100})}\n\n"

            # Get video information
            video_fps, total_frames, width, height = self.image_service.get_video_info(
                str(video_path)
            )
            skip_frames = self.image_service.calculate_skip_frames(
                video_fps, frames_per_second
            )

            # Create processed video path
            processed_video_filename = f"processed_{int(time.time())}_{filename}"
            processed_video_path = (
                Path(self.config.static_dir) / processed_video_filename
            )

            # Create session ID for counting
            session_id = f"video_{int(time.time())}_{filename.replace('.', '_')}"
            self.counting_service.reset_session(session_id)

            # Generate unique random identifier for this video processing session
            frame_prefix = secrets.token_hex(8)

            # Process video frames (this will handle cleanup of video_path in its finally block)
            async for frame_data in self._generate_video_frames(
                str(video_path),
                str(processed_video_path),
                skip_frames,
                confidence_threshold,
                session_id,
                frame_prefix,
                create_video,
                enable_classification,
            ):
                yield frame_data

        except httpx.TimeoutException:
            error_msg = "Timeout while downloading video from URL"
            print(f"[DOWNLOAD ERROR] {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        except httpx.RequestError as e:
            error_msg = f"Error downloading video from URL: {str(e)}"
            print(f"[DOWNLOAD ERROR] {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        except Exception as e:
            error_msg = f"Error processing video from URL: {str(e)}"
            print(f"[DOWNLOAD ERROR] {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        finally:
            # Clean up video file if it still exists (in case processing didn't complete)
            if video_path and video_path.exists():
                try:
                    print(f"[CLEANUP] Deleting original video file: {video_path.name}")
                    os.unlink(video_path)
                    print(f"[CLEANUP] Successfully deleted original video file")
                except Exception as e:
                    print(
                        f"[CLEANUP WARNING] Failed to delete original video file: {str(e)}"
                    )

    async def detect_video(
        self,
        file_content: bytes,
        filename: str,
        frames_per_second: int,
        confidence_threshold: float,
        create_video: bool = False,
        enable_classification: bool = False,
    ) -> StreamingResponse:
        """Detect logos in video and stream results"""
        try:
            # Save video to permanent location
            video_filename = f"uploaded_{int(time.time())}_{filename}"
            video_path = Path(self.config.static_dir) / video_filename

            with open(video_path, "wb") as f:
                f.write(file_content)

            # Get video information
            video_fps, total_frames, width, height = self.image_service.get_video_info(
                str(video_path)
            )
            skip_frames = self.image_service.calculate_skip_frames(
                video_fps, frames_per_second
            )

            # Create processed video path
            processed_video_filename = f"processed_{int(time.time())}_{filename}"
            processed_video_path = (
                Path(self.config.static_dir) / processed_video_filename
            )

            # Create session ID for counting
            session_id = f"video_{int(time.time())}_{filename.replace('.', '_')}"
            self.counting_service.reset_session(session_id)

            # Generate unique random identifier for this video processing session
            # This ensures frame names are unique even if multiple videos are processed simultaneously
            frame_prefix = secrets.token_hex(8)  # 16 character hex string

            # Create streaming response
            return StreamingResponse(
                self._generate_video_frames(
                    str(video_path),
                    str(processed_video_path),
                    skip_frames,
                    confidence_threshold,
                    session_id,
                    frame_prefix,
                    create_video,
                    enable_classification,
                ),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        except Exception as e:
            # Clean up video file
            if "video_path" in locals():
                try:
                    os.unlink(video_path)
                except:
                    pass

            raise HTTPException(
                status_code=500, detail=f"Error processing video: {str(e)}"
            )

    async def _generate_video_frames(
        self,
        video_path: str,
        processed_video_path: str,
        skip_frames: int,
        confidence_threshold: float,
        session_id: str,
        frame_prefix: str,
        create_video: bool = False,
        enable_classification: bool = False,
    ) -> AsyncGenerator[str, None]:
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
        estimated_processed_frames = (
            total_video_frames // skip_frames if skip_frames > 0 else total_video_frames
        )

        # Store detection results for interpolation
        detection_results = {}  # frame_number -> (detections, annotated_frame)

        try:
            # Send initial status with estimated total frames
            print(f"[PROCESSING] Starting video processing: {Path(video_path).name}")
            print(
                f"[PROCESSING] Total frames: {total_video_frames}, Processing: {estimated_processed_frames} frames at {skip_frames}x interval"
            )
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting video processing...', 'estimated_total_frames': estimated_processed_frames})}\n\n"

            # First pass: Process frames at specified interval and store results
            # Batch frames for GPU processing (optimize for GPU memory)
            batch_size = 8 if self.model_service.device == "cuda" else 4  # Larger batch for CUDA
            if self.model_service.device == "cuda":
                print(f"[PROCESSING] Using batch size: {batch_size} for CUDA GPU acceleration")
            frame_batch = []
            frame_indices = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frames at specified interval
                if frame_count % skip_frames == 0:
                    frame_batch.append(frame)
                    frame_indices.append(frame_count)
                    
                    # Process batch when full or at end
                    if len(frame_batch) >= batch_size:
                        # Process batch
                        for batch_frame, idx in zip(frame_batch, frame_indices):
                            try:
                                # Run detection on frame
                                detections, annotated_frame = (
                                    self.model_service.detect_in_frame(
                                        batch_frame, confidence_threshold
                                    )
                                )

                                if annotated_frame is not None:
                                    # Run classification on detections if enabled
                                    if enable_classification and self.classification_service.is_model_loaded():
                                        for detection in detections:
                                            classification_results = self._classify_detection(
                                                batch_frame, detection
                                            )
                                            if classification_results:
                                                detection.classification = classification_results
                                    
                                    # Store detection results for this frame
                                    detection_results[idx] = (
                                        detections,
                                        annotated_frame,
                                    )

                                    # Process frame for counting
                                    timestamp = idx / cap.get(cv2.CAP_PROP_FPS)
                                    frame_logo_counts = (
                                        self.counting_service.process_frame_detections(
                                            session_id, processed_count, detections, timestamp
                                        )
                                    )

                                    # Save frame to static directory for frontend display
                                    frame_filename = (
                                        f"frame_{frame_prefix}_{processed_count:06d}.jpg"
                                    )
                                    frame_path = Path(self.config.frames_dir) / frame_filename
                                    cv2.imwrite(str(frame_path), annotated_frame)
                                    frame_path.touch()

                                    frame_url = f"/static/frames/{frame_filename}"

                                    # Create frame data with counting information
                                    frame_data = {
                                        "frame_number": processed_count,
                                        "frame_url": frame_url,
                                        "detections": [
                                            detection.dict() for detection in detections
                                        ],
                                        "total_detections": len(detections),
                                        "timestamp": timestamp,
                                        "logo_counts": frame_logo_counts,
                                        "session_summary": self.counting_service.get_session_summary(
                                            session_id
                                        ),
                                    }

                                    # Send frame data
                                    yield f"data: {json.dumps({'type': 'frame', **frame_data})}\n\n"
                                    processed_count += 1

                            except Exception as e:
                                print(f"Error processing frame {idx}: {str(e)}")
                                # Store original frame if detection failed
                                detection_results[idx] = ([], batch_frame)
                        
                        # Clear batch
                        frame_batch = []
                        frame_indices = []
                        # Small async yield to prevent blocking
                        await asyncio.sleep(0.001)

                frame_count += 1
            
            # Process remaining frames in batch
            if frame_batch:
                for batch_frame, idx in zip(frame_batch, frame_indices):
                    try:
                        detections, annotated_frame = (
                            self.model_service.detect_in_frame(
                                batch_frame, confidence_threshold
                            )
                        )

                        if annotated_frame is not None:
                            # Run classification on detections if enabled
                            if enable_classification and self.classification_service.is_model_loaded():
                                for detection in detections:
                                    classification_results = self._classify_detection(
                                        batch_frame, detection
                                    )
                                    if classification_results:
                                        detection.classification = classification_results
                            
                            detection_results[idx] = (detections, annotated_frame)
                            timestamp = idx / cap.get(cv2.CAP_PROP_FPS)
                            frame_logo_counts = (
                                self.counting_service.process_frame_detections(
                                    session_id, processed_count, detections, timestamp
                                )
                            )
                            frame_filename = f"frame_{frame_prefix}_{processed_count:06d}.jpg"
                            frame_path = Path(self.config.frames_dir) / frame_filename
                            cv2.imwrite(str(frame_path), annotated_frame)
                            frame_path.touch()
                            frame_url = f"/static/frames/{frame_filename}"
                            frame_data = {
                                "frame_number": processed_count,
                                "frame_url": frame_url,
                                "detections": [detection.dict() for detection in detections],
                                "total_detections": len(detections),
                                "timestamp": timestamp,
                                "logo_counts": frame_logo_counts,
                                "session_summary": self.counting_service.get_session_summary(session_id),
                            }
                            yield f"data: {json.dumps({'type': 'frame', **frame_data})}\n\n"
                            processed_count += 1
                    except Exception as e:
                        print(f"Error processing frame {idx}: {str(e)}")
                        detection_results[idx] = ([], batch_frame)

            # Finalize real-time CSV files
            self.counting_service.finalize_session_csv_files(session_id)

            # Send completion message immediately after detection phase
            print(
                f"[PROCESSING] Detection phase complete: {processed_count} frames processed"
            )
            processed_video_url = f"/static/{Path(processed_video_path).name}" if create_video else None
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Video processing completed', 'total_frames': processed_count, 'processed_video_url': processed_video_url})}\n\n"

            # Only create video if requested
            if create_video:
                # Run video creation in background to avoid blocking
                # Note: cap is kept open for background task, will be released there
                asyncio.create_task(self._create_video_background(
                    video_path, processed_video_path, temp_frames_dir, 
                    detection_results, fps, frame_prefix, processed_video_url
                ))
            else:
                # Clean up temp frames directory if not creating video
                import shutil
                if temp_frames_dir.exists():
                    shutil.rmtree(temp_frames_dir, ignore_errors=True)
                cap.release()
                print(f"[PROCESSING] Video creation skipped (create_video=False)")

        finally:
            # Only release if not creating video (video creation handles its own release)
            if not create_video and cap.isOpened():
                cap.release()
            # Clean up original video file after processing is complete
            # Note: For video creation, cleanup happens in background task
            if not create_video and video_path and os.path.exists(video_path):
                try:
                    print(
                        f"[CLEANUP] Deleting original video file: {Path(video_path).name}"
                    )
                    os.unlink(video_path)
                    print(f"[CLEANUP] Successfully deleted original video file")
                except Exception as e:
                    print(
                        f"[CLEANUP WARNING] Failed to delete original video file: {str(e)}"
                    )

    async def _create_video_background(
        self,
        video_path: str,
        processed_video_path: str,
        temp_frames_dir: Path,
        detection_results: dict,
        fps: int,
        frame_prefix: str,
        processed_video_url: str,
    ) -> None:
        """Create video in background without blocking the main response"""
        cap = None
        try:
            print(f"[VIDEO CREATION] Starting background video creation...")
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            # Second pass: Create consistent video with interpolated detections
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Find the nearest processed frame for interpolation
                nearest_frame = self._find_nearest_processed_frame(
                    frame_count, detection_results.keys()
                )

                if nearest_frame in detection_results:
                    detections, annotated_frame = detection_results[nearest_frame]

                    # Apply the same detections to current frame
                    if detections:
                        # Create annotated frame with same detections
                        annotated_frame = self._apply_detections_to_frame(
                            frame, detections
                        )
                    else:
                        annotated_frame = frame
                else:
                    # No nearby detections, use original frame
                    annotated_frame = frame

                # Save frame to temp directory for video creation
                frame_filename = f"frame_{frame_prefix}_{frame_count:06d}.jpg"
                temp_frame_path = temp_frames_dir / frame_filename
                cv2.imwrite(str(temp_frame_path), annotated_frame)
                temp_frame_path.touch()

                frame_count += 1

            cap.release()

            # Use FFmpeg to create processed video from frames
            print(f"[VIDEO CREATION] Creating processed video with {frame_count} frames...")
            await self._create_video_from_frames(
                temp_frames_dir, processed_video_path, fps, frame_count, frame_prefix
            )

            # Clean up temp frames directory
            import shutil
            print(f"[VIDEO CREATION] Cleaning up temporary frames...")
            shutil.rmtree(temp_frames_dir, ignore_errors=True)

            print(f"[VIDEO CREATION] Video creation complete: {Path(processed_video_path).name}")
            
            # Clean up original video file after video creation is complete
            if video_path and os.path.exists(video_path):
                try:
                    print(f"[CLEANUP] Deleting original video file: {Path(video_path).name}")
                    os.unlink(video_path)
                    print(f"[CLEANUP] Successfully deleted original video file")
                except Exception as e:
                    print(f"[CLEANUP WARNING] Failed to delete original video file: {str(e)}")
        except Exception as e:
            print(f"[VIDEO CREATION ERROR] Failed to create video: {str(e)}")
            import shutil
            if temp_frames_dir.exists():
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
        finally:
            if cap is not None:
                cap.release()

    async def _create_video_from_frames(
        self,
        frames_dir: Path,
        output_path: str,
        fps: int,
        total_frames: int,
        frame_prefix: str,
    ) -> None:
        """Create MP4 video from frames using FFmpeg"""
        try:
            # FFmpeg command to create MP4 video from frames
            # -y: overwrite output file
            # -framerate: set input frame rate
            # -i: input pattern for frames (with random prefix to avoid conflicts)
            # -c:v libx264: use H.264 codec
            # -preset fast: encoding preset for speed
            # -crf 23: constant rate factor for quality
            # -pix_fmt yuv420p: pixel format for compatibility
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / f"frame_{frame_prefix}_%06d.jpg"),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",  # Optimize for web streaming
                output_path,
            ]

            # Run FFmpeg command
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                print(f"FFmpeg error: {stderr.decode()}")
                raise Exception(f"FFmpeg failed to create video: {stderr.decode()}")

            print(f"Successfully created processed video: {output_path}")

        except FileNotFoundError:
            raise Exception(
                "FFmpeg not found. Please install FFmpeg to process videos."
            )
        except Exception as e:
            raise Exception(f"Error creating video with FFmpeg: {str(e)}")

    def _find_nearest_processed_frame(
        self, current_frame: int, processed_frames: list
    ) -> int:
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

    def _apply_detections_to_frame(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
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
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return annotated_frame

    def get_session_summary(self, session_id: str) -> dict:
        """Get summary of detection session"""
        return self.counting_service.get_session_summary(session_id)

    def export_session_to_csv(
        self, session_id: str, filename_prefix: str = None
    ) -> dict:
        """Export session data to CSV files"""
        return self.counting_service.export_to_csv(session_id, filename_prefix)

    def get_available_csv_files(self) -> list:
        """Get list of available CSV files"""
        return self.counting_service.get_available_csv_files()

    def cleanup_old_csv_files(self, max_files: int = 50):
        """Clean up old CSV files"""
        self.counting_service.cleanup_old_files(max_files)

    def get_realtime_csv_files(self, session_id: str) -> dict:
        """Get real-time CSV files for a session"""
        return self.counting_service.get_realtime_csv_files(session_id)

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
