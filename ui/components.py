import streamlit as st
import torch
from utils.system_monitor import check_memory_usage

def render_system_status():
    """Render system status indicators"""
    memory_ok, memory_percent = check_memory_usage()
    gpu_available = torch.cuda.is_available()
    
    col1, col2 = st.columns(2)
    with col1:
        status_color = "green" if memory_ok else "orange"
        st.markdown(f'<div class="system-status" style="border-color: {status_color};">Memory: {memory_percent:.1f}%</div>', unsafe_allow_html=True)
    with col2:
        gpu_color = "green" if gpu_available else "gray"
        st.markdown(f'<div class="system-status" style="border-color: {gpu_color};">GPU: {"Available" if gpu_available else "CPU Only"}</div>', unsafe_allow_html=True)

def render_video_info():
    """Render current video information"""
    st.markdown("""
        <div style="margin: 2rem 0;">
            <h3 style="color: #ff8c00; text-align: center; margin-bottom: 1rem;">Current Video</h3>
        </div>
    """, unsafe_allow_html=True)
    
    video_title = st.session_state.video_title or "Unknown"
    source_text = "YouTube" if st.session_state.video_source_type == "youtube" else "Uploaded"
    
    st.markdown(f"""
        <div class="video-info">
            <strong style="color: #ff8c00;">Title:</strong> {video_title}<br>
            <strong style="color: #ff8c00;">Source:</strong> {source_text}<br>
        </div>
    """, unsafe_allow_html=True)

def render_analysis_results(analysis):
    """Render video analysis results"""
    st.markdown("""
        <div class="main-title">
            Analysis Results
        </div>
    """, unsafe_allow_html=True)
    
    duration = analysis.get('duration', 0)
    source = analysis.get('source', 'unknown')
    total_frames = analysis.get('total_frames_analyzed', 0)
    
    efficiency_metrics = analysis.get('efficiency_metrics', {})
    
    if efficiency_metrics:
        efficiency_ratio = efficiency_metrics.get('efficiency_ratio_percent', 0)
        analysis_time = efficiency_metrics.get('analysis_time_seconds', 0)
        success_rate = efficiency_metrics.get('analysis_success_rate', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **📹 Video Metrics**
            - Duration: {duration:.1f}s
            - Source: {source}
            - Frames analyzed: {total_frames}
            - Efficiency: {efficiency_ratio:.1f}%
            """)
        with col2:
            st.info(f"""
            **⚡ Performance Metrics**
            - Analysis time: {analysis_time:.1f}s
            - Success rate: {success_rate:.1f}%
            """)
    else:
        st.info(f"Duration: {duration:.1f}s | Source: {source} | Frames analyzed: {total_frames}")
    
    summary = analysis.get('summary', '')
    if summary:
        st.markdown("### Video Summary")
        st.markdown(summary)
    
    with st.expander("📋 Strategic Frame Analysis (Key Moments)"):
        frame_analyses = analysis.get('frame_analyses', [])
        for i, frame_info in enumerate(frame_analyses):
            timestamp = frame_info.get('timestamp', 0)
            description = frame_info.get('description', '')
            frame_idx = frame_info.get('frame_idx', 'N/A')
            st.markdown(f"**Frame {i+1} - {timestamp:.1f}s (#{frame_idx}):** {description}")
            if i < len(frame_analyses) - 1:
                st.markdown("---")