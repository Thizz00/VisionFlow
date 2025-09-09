# VisionFlow

Advanced video analysis framework powered by Apple's FastVLM for intelligent content understanding and conversational AI interactions.

![streamlit](/docs/streamlit.gif)

## Overview

VisionFlow provides comprehensive video analysis capabilities through intelligent frame sampling, multimodal AI processing, and context-aware chat functionality. The system processes both YouTube videos and local files, extracting detailed visual information and enabling natural language queries about video content.

## Key Features

- **Multi-Source Video Input**: YouTube URL downloading and local MP4 file uploads
- **Intelligent Frame Sampling**: Content-aware scene detection with adaptive temporal optimization
- **Vision-Language Analysis**: Detailed frame-by-frame analysis using Apple's FastVLM architecture
- **Interactive Chat Interface**: Context-preserving conversational AI for video content queries
- **Memory Management**: Real-time system monitoring with automatic resource optimization
- **Modular Architecture**: Clean separation of concerns with extensible component design

## FastVLM Model Performance Comparison

| Benchmark | FastVLM-0.5B | FastVLM-1.5B | FastVLM-7B |
|-----------|--------------|--------------|------------|
| Ai2D | 68.0 | 77.4 | 83.6 |
| ScienceQA | 85.2 | 94.4 | 96.7 |
| MMMU | 33.9 | 37.8 | 45.4 |
| VQAv2 | 76.3 | 79.1 | 80.8 |
| ChartQA | 76.0 | 80.1 | 85.0 |
| TextVQA | 64.5 | 70.4 | 74.9 |
| InfoVQA | 46.4 | 59.7 | 75.8 |
| DocVQA | 82.5 | 88.3 | 93.2 |
| OCRBench | 63.9 | 70.2 | 73.1 |
| RealWorldQA | 56.1 | 61.2 | 67.2 |
| SeedBench-Img | 71.0 | 74.2 | 75.4 |

*Source: Apple HuggingFace Model Hub*

The application uses **FastVLM-1.5B** as the default model, providing strong performance across vision-language tasks while maintaining reasonable hardware requirements. Development and testing were conducted on RTX 2060 (6GB VRAM) with 16GB system RAM.

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- min. 16GB system RAM 
- min. 6GB GPU VRAM (If CUDA)

### Setup Instructions

```bash
# Clone repository
git clone https://github.com/Thizz00/VisionFlow.git
cd visionflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run main.py
```

## Project Architecture

```
visionflow/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── main.py                     # Main Streamlit application entry point
│
├── config/
│   └── settings.py             # Configuration parameters and model settings
│
├── core/                       # Core processing modules
│   ├── __init__.py
│   ├── model_loader.py         # FastVLM model initialization and caching
│   ├── video_processor.py      # Main video analysis pipeline
│   ├── frame_analyzer.py       # Frame sampling and analysis algorithms
│   └── chat_engine.py          # Conversational AI and context management
│
├── utils/                      # Utility functions and helpers
│   ├── __init__.py
│   ├── validators.py           # Input validation for videos and URLs
│   ├── file_manager.py         # Temporary file handling and cleanup
│   ├── system_monitor.py       # Memory and GPU monitoring
│   └── logger.py              # Logging configuration and management
│
├── ui/                        # User interface components
│   ├── __init__.py
│   ├── styles.py              # CSS styling and theme definitions
│   └── components.py          # Reusable UI components and layouts
│
├── services/                  # External service integrations
│   ├── __init__.py
│   └── youtube_downloader.py   # YouTube video downloading and validation
│
└── logs/                      # Application logs (auto-generated)
    └── video_analyzer_YYYYMMDD.log
```

## Core Algorithms

### Frame Sampling Strategy
The system implements adaptive frame sampling based on video duration:
- **Short videos** (≤5s): 2 fps sampling for maximum detail
- **Medium videos** (5s-60s): Progressive scaling from 1-0.5 fps
- **Long videos** (>60s): Intelligent scene-change detection with 0.1-0.03 fps
- **Maximum limit**: 200 frames regardless of video length

### Content-Aware Scene Detection
- HSV color histogram analysis (170-dimensional feature vectors)
- Sobel gradient texture features (100-dimensional edge descriptors)
- Cosine similarity threshold (0.8) for scene change detection
- Temporal optimization with frame deduplication

## Usage

1. **Start Application**
   ```bash
   streamlit run main.py
   ```

2. **Select Input Method**
   - **YouTube Tab**: Enter video URL (max 1000 seconds)
   - **Upload Tab**: Select local MP4 file

3. **Process Video**
   - Click "Start Analysis" to begin frame extraction and analysis
   - Monitor progress and system resources during processing

4. **Interactive Chat**
   - Ask questions about video content, objects, activities, or specific timestamps
   - Receive context-aware responses based on analyzed frames

### Example Queries
```
"What objects are visible in the video?"
"Describe what happens at 2:30"
"Are there any text or signs shown?"
"How many people appear in the scene?"
"What is the main activity taking place?"
```

## Configuration

Key parameters can be adjusted in `config/settings.py`:

```python
# Model Configuration
MODEL_ID = "apple/FastVLM-1.5B"
MAX_VIDEO_DURATION = 1000  # seconds

# Processing Limits
ABSOLUTE_MAX_FRAMES = 200
SIMILARITY_THRESHOLD = 0.8

# Memory Management  
MEMORY_THRESHOLD = 85.0  # percentage
MAX_CONTEXT_FRAMES = 150
```

