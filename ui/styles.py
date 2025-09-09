import streamlit as st

def apply_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
                
        .main .block-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background-color: #0a0a0a;
            color: #ffffff;
        }
        
        .main, .block-container, .element-container {
            background-color: #0a0a0a !important;
            color: #ffffff !important;
        }
        
        .main-title {
            background: linear-gradient(135deg, #ff8c00 0%, #ff4500 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 3rem;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #ff8c00 0%, #ff4500 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 140, 0, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255, 140, 0, 0.6);
            background: linear-gradient(135deg, #ff7700 0%, #ff3300 100%);
        }
        
        .upload-area {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border: 2px dashed #ff8c00;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            transition: all 0.3s ease;
            margin: 2rem 0;
        }
        
        .video-container {
            background: #000;
            border: 2px solid #ff8c00;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(255, 140, 0, 0.3);
            margin: 2rem 0;
        }
        
        .youtube-section {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border: 2px solid #ff4500;
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0;
        }
        
        .stTextInput > div > div > input {
            border-radius: 12px;
            border: 2px solid #333;
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #ff8c00;
            box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.2);
        }
        
        .video-info {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border: 1px solid #ff8c00;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
        }
        
        .system-status {
            background: rgba(255, 140, 0, 0.1);
            border: 1px solid #ff8c00;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)