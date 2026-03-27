"""
Temporary file handler for form filling operations.
Creates temporary DOCX files from bytes for form detection and filling.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import shutil

logger = logging.getLogger(__name__)


class TemporaryFileManager:
    """Manages temporary files for form filling operations."""
    
    _temp_dir: Optional[Path] = None
    _temp_files: list = []
    
    @classmethod
    def create_temp_file(cls, file_bytes: bytes, original_filename: str) -> str:
        """
        Create a temporary file from bytes and return its path.
        
        Args:
            file_bytes: File content as bytes
            original_filename: Original filename to preserve extension
        
        Returns:
            str: Path to temporary file
        """
        try:
            # Create temp directory if not exists
            if cls._temp_dir is None:
                cls._temp_dir = Path(tempfile.gettempdir()) / "telematics_proposals"
                cls._temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract file extension
            file_ext = Path(original_filename).suffix or ".docx"
            
            # Create temporary file
            temp_file = cls._temp_dir / f"tender_{len(cls._temp_files)}{file_ext}"
            
            # Write bytes to file
            with open(temp_file, 'wb') as f:
                f.write(file_bytes)
            
            # Track for cleanup
            cls._temp_files.append(str(temp_file))
            
            logger.info(f"Created temporary file: {temp_file}")
            return str(temp_file)
        
        except Exception as e:
            logger.error(f"Failed to create temporary file: {e}")
            raise
    
    @classmethod
    def cleanup_temp_files(cls) -> None:
        """Clean up all temporary files."""
        try:
            for temp_file_path in cls._temp_files:
                temp_file = Path(temp_file_path)
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Deleted temporary file: {temp_file}")
            
            # Clean up temp directory if empty
            if cls._temp_dir and cls._temp_dir.exists():
                try:
                    if not list(cls._temp_dir.iterdir()):
                        cls._temp_dir.rmdir()
                        logger.info(f"Deleted empty temp directory: {cls._temp_dir}")
                except:
                    pass
            
            cls._temp_files = []
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    @classmethod
    def cleanup_file(cls, file_path: str) -> None:
        """Clean up a specific temporary file."""
        try:
            temp_file = Path(file_path)
            if temp_file.exists() and str(temp_file) in cls._temp_files:
                temp_file.unlink()
                cls._temp_files.remove(str(temp_file))
                logger.info(f"Deleted temporary file: {temp_file}")
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
