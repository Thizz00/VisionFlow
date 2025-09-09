import logging
import os
from datetime import datetime
from utils.system_monitor import check_memory_usage
import torch

def setup_logging():
    """Setup logging to file with proper formatting"""
    log_filename = f"video_analyzer_{datetime.now().strftime('%Y%m%d')}.log"
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Video Analyzer started")
    
    memory_ok, memory_percent = check_memory_usage()
    gpu_available = torch.cuda.is_available()
    logger.info(f"System info - Memory: {memory_percent:.1f}%, GPU: {gpu_available}")
    
    return logger, log_filename