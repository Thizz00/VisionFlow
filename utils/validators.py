import os
import cv2
import re
from typing import Tuple

def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """Validate video file before processing"""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    if os.path.getsize(file_path) == 0:
        return False, "Empty file"
    
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            return False, "Cannot open video file"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if frame_count == 0:
            return False, "No frames found in video"
        
        if fps <= 0:
            return False, "Invalid frame rate"
        
        return True, "Valid video file"
    except Exception as e:
        return False, f"Video validation error: {str(e)}"

def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL format"""
    youtube_patterns = [
        r'^https?://(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'^https?://(www\.)?youtu\.be/[\w-]+',
        r'^https?://(www\.)?youtube\.com/embed/[\w-]+'
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)