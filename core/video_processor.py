import cv2
import time
import streamlit as st
from config.settings import Config
from core.frame_analyzer import optimal_frame_sampling, analyze_frame
from utils.validators import validate_video_file
from utils.system_monitor import check_memory_usage
from utils.logger import setup_logging
import torch

logger, _ = setup_logging()

def generate_text_response(prompt, tokenizer, model):
    """Generate text response with improved parameters and memory management"""
    try:
        model.eval()
        device = next(model.parameters()).device
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        input_ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        
        with torch.no_grad():
            out = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                temperature=Config.MODEL_PARAMS['TEMPERATURE'],
                max_new_tokens=Config.MODEL_PARAMS['MAX_NEW_TOKENS_CHAT'],
                do_sample=Config.MODEL_PARAMS['DO_SAMPLE'],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=Config.MODEL_PARAMS['USE_CACHE'],
                repetition_penalty=Config.MODEL_PARAMS['REPETITION_PENALTY'],
                no_repeat_ngram_size=Config.MODEL_PARAMS['NO_REPEAT_NGRAM_SIZE']
            )
        
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Text generation error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Text generation error: {str(e)}"

def process_video(video_path, tokenizer, model, image_token_index, video_source="uploaded"):
    """Video analysis with enhanced error handling and progress tracking"""
    try:
        logger.info(f"Starting video analysis: {video_path}")
        
        is_valid, validation_msg = validate_video_file(video_path)
        if not is_valid:
            logger.error(f"Video validation failed: {validation_msg}")
            return {"error": f"Video validation failed: {validation_msg}"}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file")
            return {"error": "Could not open video file"}
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {frame_count} frames, {fps} fps, {duration:.1f}s duration")
        
        selected_frames = optimal_frame_sampling(video_path)
        
        if not selected_frames:
            logger.error("Frame sampling failed")
            cap.release()
            return {"error": "Frame sampling failed"}
        
        logger.info(f"Selected {len(selected_frames)} frames for analysis (efficiency ratio: {len(selected_frames)/frame_count*100:.1f}%)")
        
        frame_descriptions = []
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, frame_idx in enumerate(selected_frames):
            elapsed = time.time() - start_time
            if i > 0:
                rate = i / elapsed
                eta = (len(selected_frames) - i) / rate if rate > 0 else 0
                progress_info = f"Analyzing frame {i+1}/{len(selected_frames)} | Rate: {rate:.1f}/s | ETA: {eta:.1f}s"
            else:
                progress_info = f"Analyzing frame {i+1}/{len(selected_frames)}"
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None and frame.size > 0:
                timestamp = frame_idx / fps if fps > 0 else i
                
                status_text.markdown(f"""
                    <div style="text-align: center; color: #ff8c00;">
                        {progress_info}
                    </div>
                """, unsafe_allow_html=True)
                progress_bar.progress((i + 1) / len(selected_frames))
                
                logger.info(f"Analyzing frame {i+1}/{len(selected_frames)} at {timestamp:.1f}s")
                
                if i % 10 == 0:
                    memory_ok, memory_percent = check_memory_usage()
                    if not memory_ok:
                        logger.warning(f"High memory usage detected: {memory_percent}%")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                description = analyze_frame(
                    frame, tokenizer, model, image_token_index
                )
                
                if not description.startswith("Error:") and not description.startswith("Analysis error:"):
                    frame_descriptions.append({
                        "timestamp": timestamp,
                        "description": description,
                        "frame_idx": frame_idx
                    })
                    logger.info(f"Frame {i+1} analyzed successfully")
                else:
                    logger.warning(f"Frame {i+1} analysis failed: {description}")
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        if not frame_descriptions:
            logger.error("No frames could be analyzed")
            return {"error": "Could not analyze any frames from the video"}
        
        logger.info("Starting optimized summary generation")
        
        frame_text = ""
        max_chars_per_frame = 300 if len(frame_descriptions) > 100 else 500
        
        for i, frame_info in enumerate(frame_descriptions):
            timestamp = frame_info['timestamp']
            description = frame_info['description']
            
            if len(description) > max_chars_per_frame:
                description = description[:max_chars_per_frame] + "..."
            
            frame_text += f"Frame {i+1} at {timestamp:.1f}s: {description}\n\n"
        
        summary_prompt = f"""Analyze this video based on the strategically selected key frames below:

{frame_text}

Create a comprehensive summary that captures:

1. **Main Content**: What is this video about? Primary purpose and theme?

2. **Key Timeline**: Major events and how the video progresses chronologically?

3. **Visual Details**: Settings, objects, vehicles, people, and important visual elements?

4. **Specific Information**: Any text, brands, models, or identifying details visible?

5. **Context**: Location type, activities, and overall scene characteristics?

Focus on the most important and distinctive elements. Be specific about details that would help identify or understand key aspects of the video."""

        summary = generate_text_response(summary_prompt, tokenizer, model)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        efficiency_ratio = len(selected_frames) / frame_count * 100
        time_per_frame = duration / len(selected_frames) if len(selected_frames) > 0 else 0
        
        video_context = {
            "duration": duration,
            "frame_count": frame_count,
            "fps": fps,
            "frame_analyses": frame_descriptions,
            "summary": summary,
            "source": video_source,
            "total_frames_analyzed": len(frame_descriptions),
            "efficiency_metrics": {
                "total_available_frames": frame_count,
                "selected_frames": len(selected_frames),
                "successful_analyses": len(frame_descriptions),
                "efficiency_ratio_percent": efficiency_ratio,
                "analysis_time_seconds": analysis_time,
                "frames_per_second_analysis": len(frame_descriptions) / analysis_time if analysis_time > 0 else 0,
                "time_per_analyzed_frame": time_per_frame,
                "analysis_success_rate": (len(frame_descriptions) / len(selected_frames)) * 100
            },
            "model_info": {
                "model": Config.MODEL_ID
            },
            "system_info": {
                "memory_usage_percent": check_memory_usage()[1],
                "gpu_available": torch.cuda.is_available(),
            }
        }
        
        logger.info(f"Analysis completed in {analysis_time:.1f}s. {len(frame_descriptions)} frames analyzed with {efficiency_ratio:.1f}% efficiency")
        return video_context
        
    except Exception as e:
        logger.error(f"Video analysis failed: {str(e)}")
        return {"error": f"Video analysis error: {str(e)}"}