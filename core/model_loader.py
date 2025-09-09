import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.settings import Config
from utils.system_monitor import check_memory_usage
from utils.logger import setup_logging

logger, _ = setup_logging()

@st.cache_resource
def load_model():
    """Load model for both vision analysis and chat"""
    try:
        logger.info(f"Loading model: {Config.MODEL_ID}")
        
        memory_ok, memory_percent = check_memory_usage()
        if not memory_ok:
            logger.warning(f"High memory usage before model loading: {memory_percent}%")
        
        tok = AutoTokenizer.from_pretrained(Config.MODEL_ID, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Using device: {device}, dtype: {dtype}")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_info = torch.cuda.get_device_properties(0)
            logger.info(f"GPU detected: {device_name}, Memory: {memory_info.total_memory / 1e9:.1f}GB")
            model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_ID,
                dtype=dtype,
                device_map="cuda",
                trust_remote_code=True,
            )
        else:
            logger.info("Using CPU - performance may be slower")
            model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_ID,
                dtype=dtype,
                trust_remote_code=True,
            )
            model = model.to(device)
        
        logger.info("Model loaded successfully")
        return tok, model, Config.IMAGE_TOKEN_INDEX
    except Exception as e:
        error_msg = f"Error loading: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)
        return None, None, None