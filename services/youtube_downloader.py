import os
import tempfile
import shutil
import yt_dlp
from utils.validators import validate_youtube_url, validate_video_file
from utils.file_manager import temp_files_to_cleanup
from utils.logger import setup_logging
from config.settings import Config

logger, _ = setup_logging()

def download_youtube_video(url, max_duration=Config.MAX_VIDEO_DURATION):
    try:
        if not validate_youtube_url(url):
            return None, None, "Invalid YouTube URL format"
        
        logger.info(f"Starting YouTube download: {url}")
        
        temp_dir = tempfile.mkdtemp()
        temp_files_to_cleanup.add(temp_dir)
        logger.info(f"Created temp directory: {temp_dir}")
        
        try:
            free_space = shutil.disk_usage(temp_dir).free
            if free_space < Config.MIN_DISK_SPACE_MB * 1024 * 1024:
                return None, None, "Insufficient disk space for download"
        except Exception:
            pass
        
        ydl_opts = {
            'format': 'best[ext=mp4][height<=720]/best[height<=720]/mp4/best',
            'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as e:
                logger.error(f"Failed to extract video info: {e}")
                return None, None, f"Cannot access video: {str(e)}"
                
            duration = info.get('duration', 0)
            title = info.get('title', 'Unknown')
            
            logger.info(f"Video info - Title: {title}, Duration: {duration}s")
            
            if duration > max_duration:
                logger.warning(f"Video too long: {duration}s (max {max_duration}s)")
                return None, None, f"Video too long: {duration}s (max {max_duration}s allowed)"
            
            ydl.download([url])
            
            video_file = None
            for file in os.listdir(temp_dir):
                if file.startswith('video.'):
                    video_file = os.path.join(temp_dir, file)
                    break
            
            if video_file and os.path.exists(video_file):
                is_valid, validation_msg = validate_video_file(video_file)
                if not is_valid:
                    return None, None, f"Downloaded file invalid: {validation_msg}"
                
                permanent_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                permanent_temp.close()
                shutil.move(video_file, permanent_temp.name)
                temp_files_to_cleanup.add(permanent_temp.name)
                
                logger.info(f"Video downloaded successfully: {permanent_temp.name}")
                return permanent_temp.name, title, f"Downloaded: {title} ({duration}s)"
            else:
                logger.error("Download failed - file not found")
                return None, None, "Download failed - file not found"
                
    except Exception as e:
        logger.error(f"YouTube download error: {str(e)}")
        return None, None, f"Download error: {str(e)}"