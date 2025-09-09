import os
import shutil

temp_files_to_cleanup = set()

def cleanup_temp_files():
    """Clean up temporary files on exit"""
    for temp_file in temp_files_to_cleanup.copy():
        try:
            if os.path.exists(temp_file):
                if os.path.isfile(temp_file):
                    os.unlink(temp_file)
                else:
                    shutil.rmtree(temp_file, ignore_errors=True)
                temp_files_to_cleanup.discard(temp_file)
        except Exception:
            pass