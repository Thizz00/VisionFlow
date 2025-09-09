import torch
import cv2
import numpy as np
from PIL import Image
import time
from config.settings import Config
from utils.system_monitor import check_memory_usage
from utils.logger import setup_logging
from sklearn.metrics.pairwise import cosine_similarity

logger, _ = setup_logging()

def extract_frame_features(frame):
    """Extract features from frame for similarity comparison"""
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (64, 64))
        
        grad_x = cv2.Sobel(gray_small, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_small, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_features = gradient_magnitude.flatten()[:100]
        
        features = np.concatenate([hist_h, hist_s, hist_v, gradient_features])
        return features
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        return np.random.rand(270)

def content_aware_sampling(video_path, target_frames=50, similarity_threshold=0.85):
    """Content-based sampling - selects frames when scene changes"""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count == 0 or fps == 0:
            cap.release()
            return []
        
        analysis_step = max(1, frame_count // 1000)
        
        logger.info(f"Content-aware sampling: analyzing every {analysis_step} frame(s)")
        
        selected_frames = [0]
        last_features = None
        
        for frame_idx in range(0, frame_count, analysis_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
                
            current_features = extract_frame_features(frame)
            
            if last_features is not None:
                similarity = cosine_similarity([last_features], [current_features])[0][0]
                
                if similarity < similarity_threshold:
                    selected_frames.append(frame_idx)
                    last_features = current_features
                    logger.debug(f"Scene change at frame {frame_idx} (similarity: {similarity:.3f})")
            else:
                last_features = current_features
        
        if selected_frames[-1] != frame_count - 1:
            selected_frames.append(frame_count - 1)
        
        cap.release()
        
        if len(selected_frames) > target_frames:
            indices = np.linspace(0, len(selected_frames)-1, target_frames, dtype=int)
            selected_frames = [selected_frames[i] for i in indices]
        elif len(selected_frames) < target_frames // 2:
            additional_frames = np.linspace(0, frame_count-1, target_frames, dtype=int)
            selected_frames = sorted(list(set(selected_frames + additional_frames.tolist())))[:target_frames]
        
        logger.info(f"Content-aware sampling selected {len(selected_frames)} frames")
        return selected_frames
        
    except Exception as e:
        logger.error(f"Content-aware sampling error: {str(e)}")
        return []

def simple_frame_sampling(video_path, target_frames=50):
    """Simple uniform sampling as fallback"""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if frame_count <= target_frames:
            return list(range(frame_count))
        else:
            return np.linspace(0, frame_count-1, target_frames, dtype=int).tolist()
    except Exception as e:
        logger.error(f"Simple sampling error: {str(e)}")
        return []

def optimal_frame_sampling(video_path):
    """Optimized and intelligent frame sampling with strict limits"""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"Optimal sampling for video: {duration:.1f}s, {frame_count} frames, {fps:.1f} fps")
        
        sampling_config = Config.FRAME_SAMPLING
        
        if duration <= sampling_config['SHORT_VIDEO_THRESHOLD']:
            target_frames = min(10, int(duration * 2))
        elif duration <= sampling_config['MEDIUM_SHORT_THRESHOLD']:
            target_frames = min(15, int(duration))
        elif duration <= sampling_config['MEDIUM_THRESHOLD']:
            target_frames = min(20, int(duration / 1.5))
        elif duration <= sampling_config['LONG_THRESHOLD']:
            target_frames = min(30, int(duration / 2))
        elif duration <= sampling_config['VERY_LONG_THRESHOLD']:
            target_frames = min(45, int(duration / 4))
        elif duration <= sampling_config['EXTREMELY_LONG_THRESHOLD']:
            target_frames = min(75, int(duration / 8))
        elif duration <= sampling_config['EXTREME_THRESHOLD']:
            target_frames = min(120, int(duration / 15))
        else:
            target_frames = min(180, int(duration / 30))
        
        target_frames = min(target_frames, sampling_config['ABSOLUTE_MAX_FRAMES'])
        
        logger.info(f"Optimal target: {target_frames} frames ({target_frames/duration:.3f} frames/s)")
        
        if target_frames < frame_count:
            return content_aware_sampling(video_path, target_frames, similarity_threshold=Config.SIMILARITY_THRESHOLD)
        else:
            return list(range(frame_count))
            
    except Exception as e:
        logger.error(f"Optimal sampling error: {str(e)}")
        return simple_frame_sampling(video_path, 50)

def analyze_frame(frame, tokenizer, model, image_token_index, retry_count=2):
    """Analyze single frame using model with comprehensive prompt and retry logic"""
    for attempt in range(retry_count + 1):
        try:
            memory_ok, memory_percent = check_memory_usage()
            if not memory_ok:
                logger.warning(f"High memory usage: {memory_percent}%")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(0.1)
            
            model.eval()
            device = next(model.parameters()).device
            
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.error("Invalid frame data")
                return "Error: Invalid frame data"
            
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.error("Invalid frame format")
                return "Error: Invalid frame format"
            
            height, width = frame.shape[:2]
            if height < 10 or width < 10:
                logger.error("Frame too small")
                return "Error: Frame too small"
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            detailed_prompt = """Provide an extremely detailed analysis of everything visible in this image. Leave nothing out and be as specific as possible:

COMPLETE VISUAL INVENTORY:
- Every object, item, structure, surface, and element visible
- All colors, shapes, sizes, textures, materials, and patterns
- Exact positions, orientations, distances, and spatial relationships
- Any text, numbers, symbols, logos, signs, or markings of any kind
- Brand names, model numbers, identifying features, or distinctive characteristics
- All lighting conditions, shadows, reflections, and visual effects

COMPREHENSIVE SCENE DESCRIPTION:
- Precise location type, environment, and setting details
- Weather conditions, time of day indicators, seasonal clues
- Architectural features, construction materials, design elements
- Ground surfaces, vegetation, natural features, terrain
- Any damage, wear, age, or condition indicators

COMPLETE ENTITY ANALYSIS:
- All living beings: appearance, clothing, posture, expressions, activities
- All manufactured items: purpose, condition, specifications, notable features  
- All natural elements: type, state, characteristics, arrangement
- Any tools, equipment, machinery, technology, or instruments
- Transportation methods, infrastructure, pathways, or routes

DETAILED ACTIVITY AND CONTEXT:
- Everything happening in the scene, no matter how minor
- Any movement, interaction, process, or ongoing activity
- Cause and effect relationships between elements
- Temporal indicators suggesting before/after states
- Any unusual, interesting, or noteworthy aspects

TECHNICAL AND QUALITY DETAILS:
- Image quality, clarity, focus, lighting conditions
- Camera angle, perspective, framing, composition
- Any visual artifacts, distortions, or technical aspects
- Color accuracy, saturation, contrast observations

Describe everything with maximum precision and specificity. Use exact measurements when possible, specific color names, proper terminology, and avoid any vague or generic descriptions."""

            messages = [
                {"role": "user", "content": f"<image>\n{detailed_prompt}"}
            ]
            
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            pre, post = rendered.split("<image>", 1)
            
            pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            
            img_tok = torch.tensor([[image_token_index]], dtype=pre_ids.dtype, device=device)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
            attention_mask = torch.ones_like(input_ids, device=device)
            
            px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
            px = px.to(device, dtype=model.dtype)
            
            with torch.no_grad():
                out = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px,
                    temperature=Config.MODEL_PARAMS['TEMPERATURE'],
                    max_new_tokens=Config.MODEL_PARAMS['MAX_NEW_TOKENS_ANALYSIS'],
                    do_sample=Config.MODEL_PARAMS['DO_SAMPLE'],
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=Config.MODEL_PARAMS['USE_CACHE'],
                    repetition_penalty=Config.MODEL_PARAMS['REPETITION_PENALTY']
                )
            
            response = tokenizer.decode(out[0], skip_special_tokens=True)
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response.strip()
            
        except Exception as e:
            if attempt < retry_count:
                logger.warning(f"Frame analysis attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(0.5)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"Frame analysis failed after {retry_count + 1} attempts: {str(e)}")
                return f"Analysis error: {str(e)}"