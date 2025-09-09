import streamlit as st
import os
import tempfile
import atexit
import torch

from core.model_loader import load_model
from core.video_processor import process_video
from core.chat_engine import smart_context_chat
from services.youtube_downloader import download_youtube_video
from utils.validators import validate_video_file, validate_youtube_url
from utils.file_manager import cleanup_temp_files, temp_files_to_cleanup
from utils.system_monitor import check_memory_usage
from utils.logger import setup_logging
from ui.styles import apply_styles
from ui.components import render_system_status, render_video_info, render_analysis_results

atexit.register(cleanup_temp_files)
logger, log_file = setup_logging()

st.set_page_config(page_title="VisionFlow", layout="wide")

if "video_analysis" not in st.session_state:
    st.session_state.video_analysis = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "video_context" not in st.session_state:
    st.session_state.video_context = ""
if "current_video_path" not in st.session_state:
    st.session_state.current_video_path = None
if "current_video_file" not in st.session_state:
    st.session_state.current_video_file = None
if "video_source_type" not in st.session_state:
    st.session_state.video_source_type = None
if "video_title" not in st.session_state:
    st.session_state.video_title = None
if "last_memory_check" not in st.session_state:
    st.session_state.last_memory_check = 0.0

def main():
    apply_styles()
    
    st.markdown("""
        <div class="main-title">
            VisionFlow
        </div>
    """, unsafe_allow_html=True)
    
    render_system_status()
    
    with st.spinner("Loading model..."):
        tokenizer, model, image_token_index = load_model()
    
    if tokenizer is None or model is None:
        st.error("Failed to load model")
        st.stop()
    
    tab1, tab2 = st.tabs(["YouTube URL", "Upload File"])

    with tab1:
        st.markdown("""
                <div class="youtube-section">
                    <h4 style="color: #ff4500; text-align: center; margin-bottom: 1rem;">YouTube Video</h4>
                    <p style="color: #cccccc; margin-bottom: 1.5rem;text-align: center;">Enter YouTube URL (max 1000 seconds)</p>
                </div>
        """, unsafe_allow_html=True)
            
        youtube_url = st.text_input(
                "YouTube URL", 
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed"
        )
        
        if youtube_url:
            if validate_youtube_url(youtube_url):
                st.success("Valid YouTube URL format")
            else:
                st.error("Invalid YouTube URL format")
            
        if youtube_url and st.button("Download YouTube Video", use_container_width=True):
            if not validate_youtube_url(youtube_url):
                st.error("Please enter a valid YouTube URL")
            else:
                try:
                    with st.spinner("Downloading from YouTube..."):
                        video_path, title, message = download_youtube_video(youtube_url, max_duration=1000)
                        
                    if video_path:
                        st.success(message)
                            
                        st.session_state.current_video_path = video_path
                        st.session_state.video_source_type = "youtube"
                        st.session_state.current_video_file = None
                        st.session_state.video_title = title
                        st.session_state.video_processed = False
                        st.session_state.video_analysis = None
                        st.session_state.chat_history = []
                            
                        st.rerun()
                    else:
                        st.error(message)
                except Exception as e:
                    error_msg = f"Download failed: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
    
    upload_container = tab2 
    
    with upload_container:
        st.markdown("""
            <div class="upload-area">
                <h4 style="color: #ff8c00; margin-bottom: 1rem;text-align: center;">Upload Video File</h4>
                <p style="color: #cccccc;text-align: center;">Upload MP4 files</p>
            </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("video_upload", type=['mp4'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.session_state.current_video_file != uploaded_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                temp_files_to_cleanup.add(tmp_file_path)
                
                is_valid, validation_msg = validate_video_file(tmp_file_path)
                if not is_valid:
                    st.error(f"Invalid video file: {validation_msg}")
                    logger.error(f"Invalid uploaded file: {validation_msg}")
                else:
                    st.success("Valid video file uploaded")
                    logger.info(f"New file uploaded and validated: {uploaded_file.name}")
                    
                    st.session_state.current_video_file = uploaded_file
                    st.session_state.current_video_path = tmp_file_path
                    st.session_state.video_source_type = "uploaded"
                    st.session_state.video_title = uploaded_file.name
                    st.session_state.video_processed = False
                    st.session_state.video_analysis = None
                    st.session_state.chat_history = []
                    
                    st.rerun()
            except Exception as e:
                error_msg = f"File upload failed: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
    
    if st.session_state.current_video_path and os.path.exists(st.session_state.current_video_path):
        st.markdown("---")
        render_video_info()
        
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        
        try:
            with open(st.session_state.current_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
        except Exception as e:
            st.error(f"Error displaying video: {str(e)}")
            logger.error(f"Video display error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state.video_processed:        
            if st.button(f"Start Analysis", use_container_width=True, key="analyze_btn"):
                is_valid, validation_msg = validate_video_file(st.session_state.current_video_path)
                if not is_valid:
                    st.error(f"Cannot analyze - {validation_msg}")
                    logger.error(f"Pre-analysis validation failed: {validation_msg}")
                else:
                    memory_ok, memory_percent = check_memory_usage()
                    if not memory_ok:
                        st.warning(f"High memory usage detected: {memory_percent:.1f}%. Analysis may be slower.")
                        logger.warning(f"Starting analysis with high memory usage: {memory_percent:.1f}%")
                    
                    with st.spinner("Running video analysis..."):
                        video_analysis = process_video(
                            st.session_state.current_video_path, 
                            tokenizer, 
                            model, 
                            image_token_index,
                            video_source=st.session_state.video_source_type
                        )
                        
                        if "error" in video_analysis:
                            st.error(f"Analysis failed: {video_analysis['error']}")
                        else:
                            st.session_state.video_analysis = video_analysis
                            st.session_state.video_processed = True
                            st.session_state.video_context = video_analysis
                            st.session_state.chat_history = []
                                                
                            st.rerun()
    
    if st.session_state.video_processed and st.session_state.video_analysis:
        st.markdown("---")
        render_analysis_results(st.session_state.video_analysis)
        
        st.markdown("---")
        st.markdown("""
            <div class="main-title">
                Smart Video Chat
            </div>
            <div style="text-align: center; margin-bottom: 2rem;">
                <p style="color: #ff8c00;">Ask questions about specific details with improved context understanding</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        if user_question := st.chat_input("Ask about the video..."):
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching frame data..."):
                    ai_response = smart_context_chat(
                        user_question, 
                        st.session_state.video_context, 
                        st.session_state.chat_history,
                        tokenizer,
                        model
                    )
                st.markdown(ai_response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                logger.info("Chat history cleared")
                st.rerun()
        with col2:
            if st.button("Free Memory", use_container_width=True):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.success("GPU memory cleared")
                logger.info("Manual memory cleanup performed")

if __name__ == "__main__":
    main()