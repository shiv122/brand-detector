import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime
import os

from models.detection import Detection


class LogoCountingService:
    """Service to handle logo counting and CSV export functionality"""

    def __init__(self, static_dir: str):
        self.static_dir = Path(static_dir)
        self.csv_dir = self.static_dir / "csv_reports"
        self.csv_dir.mkdir(exist_ok=True)

        # In-memory storage for current session counts
        self.session_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.frame_counts: Dict[str, int] = defaultdict(int)
        self.detection_history: Dict[str, List[Dict]] = defaultdict(list)

        # Real-time CSV file handles and writers
        self.csv_writers: Dict[str, Dict[str, any]] = defaultdict(dict)
        self.csv_files: Dict[str, Dict[str, str]] = defaultdict(dict)

    def reset_session(self, session_id: str):
        """Reset counting for a new session"""
        self.session_counts[session_id] = defaultdict(int)
        self.frame_counts[session_id] = 0
        self.detection_history[session_id] = []

        # Close any existing CSV files for this session
        self._close_session_csv_files(session_id)

        # Ensure only current run files exist by clearing any prior CSV files
        self._clear_csv_directory()

        # Initialize real-time CSV files
        self._initialize_realtime_csv_files(session_id)

    def process_frame_detections(
        self,
        session_id: str,
        frame_number: int,
        detections: List[Detection],
        timestamp: float = None,
    ) -> Dict[str, int]:
        """Process detections for a frame and update counts"""
        # Increment frame count
        self.frame_counts[session_id] += 1

        # Count logos in this frame
        frame_logo_counts = defaultdict(int)
        for detection in detections:
            logo_name = detection.class_name
            frame_logo_counts[logo_name] += 1
            # Add to session total
            self.session_counts[session_id][logo_name] += 1

        # Store detection history
        detection_record = {
            "frame_number": frame_number,
            "timestamp": timestamp or 0.0,
            "logo_counts": dict(frame_logo_counts),
            "total_detections": len(detections),
            "detections": [detection.dict() for detection in detections],
        }
        self.detection_history[session_id].append(detection_record)

        # Write to real-time CSV files
        self._write_to_realtime_csv(
            session_id, frame_number, timestamp, detections, frame_logo_counts
        )

        return dict(frame_logo_counts)

    def _initialize_realtime_csv_files(self, session_id: str):
        """Initialize real-time CSV files for a session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{session_id[:8]}"
        filename_prefix = f"detection_report_{unique_id}"

        # Create single CSV file with detections grouped by brand
        csv_path = self.csv_dir / f"{filename_prefix}.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "Brand",
                "Frame Number",
                "Timestamp",
                "Confidence",
                "Bounding Box",
                "Classification",
            ]
        )

        # Store file handles and writers
        self.csv_writers[session_id] = {
            "main": {"writer": csv_writer, "file": csv_file},
        }

        self.csv_files[session_id] = {
            "main": f"/static/csv_reports/{csv_path.name}",
        }

    def _write_to_realtime_csv(
        self,
        session_id: str,
        frame_number: int,
        timestamp: float,
        detections: List[Detection],
        frame_logo_counts: Dict[str, int],
    ):
        """Write detection data to real-time CSV files"""
        if session_id not in self.csv_writers:
            return

        # Write to main CSV file (one row per detection)
        main_writer = self.csv_writers[session_id]["main"]["writer"]

        for detection in detections:
            box = detection.bbox
            box_str = f"[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"
            confidence_str = f"{detection.confidence:.3f}"
            
            # Format classification results if available
            if detection.classification and len(detection.classification) > 0:
                top_class = detection.classification[0]
                classification_str = f"{top_class.class_name} ({top_class.confidence:.2%})"
                if len(detection.classification) > 1:
                    classification_str += f" | {detection.classification[1].class_name} ({detection.classification[1].confidence:.2%})"
            else:
                classification_str = "N/A"
            
            # Write one row per detection
            main_writer.writerow(
                [
                    detection.class_name,  # Brand
                    frame_number,  # Frame number (repeated for each detection)
                    f"{timestamp:.2f}",  # Timestamp (repeated for each detection)
                    confidence_str,  # Single confidence value
                    box_str,  # Single bounding box
                    classification_str,  # Classification result
                ]
            )

        # Flush to ensure data is written immediately
        self.csv_writers[session_id]["main"]["file"].flush()

    def _close_session_csv_files(self, session_id: str):
        """Close CSV files for a session"""
        if session_id in self.csv_writers:
            for csv_type, data in self.csv_writers[session_id].items():
                try:
                    data["file"].close()
                except:
                    pass
            del self.csv_writers[session_id]

        if session_id in self.csv_files:
            del self.csv_files[session_id]

    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of current session"""
        return {
            "session_id": session_id,
            "total_frames_processed": self.frame_counts[session_id],
            "logo_totals": dict(self.session_counts[session_id]),
            "total_detections": sum(self.session_counts[session_id].values()),
            "unique_logos": list(self.session_counts[session_id].keys()),
            "realtime_csv_files": self.csv_files.get(session_id, {}),
        }

    def export_to_csv(
        self, session_id: str, filename_prefix: str = None
    ) -> Dict[str, str]:
        """Export detection data to CSV files"""
        if not filename_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = f"{timestamp}_{session_id[:8]}"
            filename_prefix = f"detection_export_{unique_id}"

        # Create single CSV file with detections grouped by brand
        csv_path = self.csv_dir / f"{filename_prefix}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Brand",
                    "Frame Number",
                    "Timestamp",
                    "Count in Frame",
                    "Confidences",
                    "Bounding Boxes",
                ]
            )

            for record in self.detection_history[session_id]:
                frame_number = record["frame_number"]
                timestamp = record["timestamp"]

                # Group detections in this frame by brand
                brand_to_boxes: Dict[str, list] = defaultdict(list)
                brand_to_confidences: Dict[str, list] = defaultdict(list)
                for detection_data in record["detections"]:
                    brand_name = detection_data["class_name"]
                    bbox = detection_data["bbox"]
                    brand_to_boxes[brand_name].append(
                        f"[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]"
                    )
                    brand_to_confidences[brand_name].append(
                        f"{detection_data['confidence']:.3f}"
                    )

                for brand, boxes in brand_to_boxes.items():
                    count_in_frame = len(boxes)
                    boxes_str = ", ".join(boxes)
                    confidences_str = ", ".join(brand_to_confidences[brand])
                    writer.writerow(
                        [
                            brand,
                            frame_number,
                            f"{timestamp:.2f}",
                            count_in_frame,
                            confidences_str,
                            boxes_str,
                        ]
                    )

        return {"main": f"/static/csv_reports/{csv_path.name}"}

    def get_realtime_csv_files(self, session_id: str) -> Dict[str, str]:
        """Get real-time CSV files for a session"""
        return self.csv_files.get(session_id, {})

    def finalize_session_csv_files(self, session_id: str):
        """Finalize and close CSV files for a session"""
        if session_id in self.csv_writers:
            # Close all files
            for csv_type, data in self.csv_writers[session_id].items():
                try:
                    data["file"].close()
                except:
                    pass

            # Clean up writers
            del self.csv_writers[session_id]

    def _clear_csv_directory(self):
        """Remove all CSV files so only current run files are present."""
        try:
            for csv_file in self.csv_dir.glob("*.csv"):
                try:
                    csv_file.unlink()
                except Exception as e:
                    print(f"Error deleting CSV file {csv_file}: {e}")
        except Exception as e:
            print(f"Error clearing CSV directory {self.csv_dir}: {e}")

    def get_available_csv_files(self) -> List[Dict[str, str]]:
        """Get list of available CSV files"""
        csv_files = []
        for csv_file in self.csv_dir.glob("*.csv"):
            csv_files.append(
                {
                    "filename": csv_file.name,
                    "path": f"/static/csv_reports/{csv_file.name}",
                    "size": csv_file.stat().st_size,
                    "created": datetime.fromtimestamp(
                        csv_file.stat().st_ctime
                    ).isoformat(),
                }
            )
        return sorted(csv_files, key=lambda x: x["created"], reverse=True)

    def cleanup_old_files(self, max_files: int = 50):
        """Clean up old CSV files, keeping only the most recent ones"""
        csv_files = sorted(
            self.csv_dir.glob("*.csv"), key=lambda x: x.stat().st_ctime, reverse=True
        )

        if len(csv_files) > max_files:
            for old_file in csv_files[max_files:]:
                try:
                    old_file.unlink()
                except Exception as e:
                    print(f"Error deleting old CSV file {old_file}: {e}")
