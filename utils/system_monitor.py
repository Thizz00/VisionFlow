import psutil
from typing import Tuple
from config.settings import Config

def check_memory_usage() -> Tuple[bool, float]:
    """Check current memory usage"""
    try:
        memory = psutil.virtual_memory()
        return memory.percent < Config.MEMORY_THRESHOLD, memory.percent
    except Exception:
        return True, 0.0