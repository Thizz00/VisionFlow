import numpy as np
import torch
from config.settings import Config
from core.video_processor import generate_text_response
from utils.logger import setup_logging

logger, _ = setup_logging()

def smart_context_chat(question, video_context, chat_history, tokenizer, model):
    """Smart chat with context optimization based on question type and better memory management"""
    try:
        logger.info(f"Processing chat question: {question[:50]}...")
        
        if not video_context or not isinstance(video_context, dict):
            logger.error("No video context available for chat")
            return "Error: No video analysis available. Please analyze a video first."
        
        summary = video_context.get('summary', '')
        frame_analyses = video_context.get('frame_analyses', [])
        duration = video_context.get('duration', 0)
        
        max_context_frames = Config.MAX_CONTEXT_FRAMES
        
        if len(frame_analyses) <= max_context_frames:
            selected_context = frame_analyses
            context_note = ""
        else:
            indices = np.linspace(0, len(frame_analyses)-1, max_context_frames, dtype=int)
            selected_context = [frame_analyses[i] for i in indices]
            context_note = f"\n(Note: Using {max_context_frames} representative frames out of {len(frame_analyses)} total for optimal response speed)\n"
        
        context_text = f"Video Summary: {summary}\n\nDuration: {duration:.1f} seconds{context_note}\n"
        context_text += f"\nKey Frame Analysis ({len(selected_context)} frames):\n\n"
        
        for i, frame_info in enumerate(selected_context):
            timestamp = frame_info.get('timestamp', 0)
            description = frame_info.get('description', '')
            if len(description) > 400:
                description = description[:400] + "..."
            context_text += f"Frame {i+1} [{timestamp:.1f}s]: {description}\n\n"
        
        history_text = ""
        if chat_history and len(chat_history) > 1:
            history_text = "\n\nRecent conversation:\n"
            recent_history = chat_history[-6:-1] if len(chat_history) > 6 else chat_history[:-1]
            for msg in recent_history:
                role = msg['role']
                content = msg['content'][:150]
                history_text += f"{role}: {content}\n"
        
        chat_prompt = f"""You are analyzing a video based on strategically selected key frames. Here's the context:

{context_text}{history_text}

User question: {question}

INSTRUCTIONS:
1. Answer based ONLY on the frame descriptions provided above
2. Reference specific timestamps when relevant
3. If asked about details not visible in the frame descriptions, state that clearly
4. Be precise and specific - quote exact details from the timeline
5. Don't provide generic responses - use the actual analyzed frame data

Provide a focused answer based on the frame analysis data."""
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        response = generate_text_response(chat_prompt, tokenizer, model)
        
        logger.info("Smart context chat response generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return f"Chat error: {str(e)}"