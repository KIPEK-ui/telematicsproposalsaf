"""
Document Manager Service
Handles file uploads, duplicate detection, and document processing for RAG training.
Manages files in data/tenders_proposals/ with validation and conflict prevention.
"""

import logging
import hashlib
import shutil
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for uploaded documents."""
    filename: str
    file_size: int
    file_hash: str
    upload_date: str
    file_type: str
    status: str  # 'pending', 'processed', 'error'
    processed_path: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class UploadResult:
    """Result of a document upload."""
    success: bool
    filename: str
    file_path: str
    metadata: DocumentMetadata
    message: str
    is_duplicate: bool = False


class DocumentManagerError(Exception):
    """Raised when document manager operation fails."""
    pass


class DocumentManager:
    """
    Manages document uploads, storage, and RAG training.
    Handles duplicate detection, file validation, and processing.
    """

    def __init__(self, storage_dir: Optional[str] = None, training_dir: Optional[str] = None):
        """
        Initialize document manager.
        
        Args:
            storage_dir: Directory to store uploaded documents.
                        Defaults to data/tenders_proposals/
            training_dir: Directory to store processed training data.
                         Defaults to data/training_proposals/
        """
        self.storage_dir = Path(storage_dir or "data/tenders_proposals")
        self.training_dir = Path(training_dir or "data/training_proposals")
        
        # Create directories if they don't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file to track uploads and processing state
        self.metadata_file = self.storage_dir / ".processing_metadata.json"
        # Load existing processing state
        self._processing_metadata = self._load_processing_metadata()
        
        logger.info(f"DocumentManager initialized:")
        logger.info(f"  Storage: {self.storage_dir}")
        logger.info(f"  Training: {self.training_dir}")
        logger.info(f"  Processing state loaded: {len(self._processing_metadata)} entries")

    def calculate_file_hash(self, file_bytes: bytes) -> str:
        """
        Calculate SHA256 hash of file content.
        
        Args:
            file_bytes: File content as bytes
            
        Returns:
            str: SHA256 hash
        """
        return hashlib.sha256(file_bytes).hexdigest()

    def file_exists_by_hash(self, file_hash: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a file with this hash already exists.
        
        Args:
            file_hash: SHA256 hash of file content
            
        Returns:
            Tuple: (exists: bool, existing_filename: Optional[str])
        """
        for file_path in self.storage_dir.glob("*"):
            if file_path.is_file() and file_path.name != ".metadata.txt":
                try:
                    with open(file_path, "rb") as f:
                        existing_hash = self.calculate_file_hash(f.read())
                    if existing_hash == file_hash:
                        return True, file_path.name
                except Exception as e:
                    logger.warning(f"Could not hash {file_path}: {str(e)}")
        
        return False, None

    def file_exists_by_name(self, filename: str) -> bool:
        """
        Check if a file with this name exists.
        
        Args:
            filename: Name of the file
            
        Returns:
            bool: True if file exists
        """
        file_path = self.storage_dir / filename
        return file_path.exists()

    def upload_file(self, file_bytes: bytes, filename: str, overwrite: bool = False) -> UploadResult:
        """
        Upload a document file, detecting duplicates.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            overwrite: Whether to overwrite if file exists (default: False)
            
        Returns:
            UploadResult: Result of upload operation
        """
        try:
            # Validate filename
            if not filename or len(filename) < 3:
                return UploadResult(
                    success=False,
                    filename=filename,
                    file_path="",
                    metadata=DocumentMetadata(
                        filename=filename,
                        file_size=0,
                        file_hash="",
                        upload_date="",
                        file_type="",
                        status="error",
                        error_message="Invalid filename"
                    ),
                    message="❌ Filename is too short or invalid",
                    is_duplicate=False
                )
            
            # Check file size
            file_size = len(file_bytes)
            max_size = 50 * 1024 * 1024  # 50 MB
            if file_size > max_size:
                return UploadResult(
                    success=False,
                    filename=filename,
                    file_path="",
                    metadata=DocumentMetadata(
                        filename=filename,
                        file_size=file_size,
                        file_hash="",
                        upload_date=datetime.now().isoformat(),
                        file_type=Path(filename).suffix,
                        status="error",
                        error_message=f"File too large: {file_size / 1024 / 1024:.2f} MB (max 50 MB)"
                    ),
                    message=f"❌ File too large: {file_size / 1024 / 1024:.2f} MB (max 50 MB)",
                    is_duplicate=False
                )
            
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_bytes)
            
            # Check for duplicates by hash
            hash_exists, existing_name = self.file_exists_by_hash(file_hash)
            if hash_exists and not overwrite:
                return UploadResult(
                    success=False,
                    filename=filename,
                    file_path="",
                    metadata=DocumentMetadata(
                        filename=filename,
                        file_size=file_size,
                        file_hash=file_hash,
                        upload_date=datetime.now().isoformat(),
                        file_type=Path(filename).suffix,
                        status="pending",
                        error_message=f"Duplicate of existing file: {existing_name}"
                    ),
                    message=f"⚠️ This file already exists as '{existing_name}'",
                    is_duplicate=True
                )
            
            # Check for filename conflicts
            file_path = self.storage_dir / filename
            if file_path.exists() and not overwrite:
                # Rename with timestamp
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"{stem}_{timestamp}{suffix}"
                file_path = self.storage_dir / new_filename
                logger.info(f"Renamed file due to conflict: {filename} → {new_filename}")
            
            # Write file
            file_path.write_bytes(file_bytes)
            
            # Create metadata
            metadata = DocumentMetadata(
                filename=file_path.name,
                file_size=file_size,
                file_hash=file_hash,
                upload_date=datetime.now().isoformat(),
                file_type=Path(filename).suffix.lstrip('.').lower(),
                status="pending"
            )
            
            logger.info(f"File uploaded: {filename} → {file_path.name}")
            
            return UploadResult(
                success=True,
                filename=file_path.name,
                file_path=str(file_path),
                metadata=metadata,
                message=f"✅ File '{filename}' uploaded successfully",
                is_duplicate=False
            )
            
        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            logger.error(error_msg)
            return UploadResult(
                success=False,
                filename=filename,
                file_path="",
                metadata=DocumentMetadata(
                    filename=filename,
                    file_size=len(file_bytes),
                    file_hash="",
                    upload_date=datetime.now().isoformat(),
                    file_type=Path(filename).suffix,
                    status="error",
                    error_message=str(e)
                ),
                message=f"❌ {error_msg}",
                is_duplicate=False
            )

    def list_uploaded_documents(self) -> List[Dict[str, Any]]:
        """
        List all uploaded documents with metadata.
        Checks persistent processing metadata (JSON file).
        
        Returns:
            List of document info dictionaries
        """
        documents = []
        
        for file_path in sorted(self.storage_dir.glob("*")):
            if file_path.is_file() and file_path.name not in [".processing_metadata.json", ".metadata.txt"]:
                try:
                    stat = file_path.stat()
                    file_hash = self.calculate_file_hash(file_path.read_bytes())
                    
                    # Check processing state from persistent metadata (JSON)
                    processed = self._processing_metadata.get(file_path.name, {}).get('processed', False)
                    
                    documents.append({
                        "filename": file_path.name,
                        "size_mb": round(stat.st_size / 1024 / 1024, 2),
                        "type": file_path.suffix.lstrip('.').upper() or "UNKNOWN",
                        "uploaded_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "hash": file_hash[:8] + "...",
                        "processed": processed,
                        "path": str(file_path)
                    })
                except Exception as e:
                    logger.warning(f"Could not list document {file_path}: {str(e)}")
        
        return documents

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about uploaded documents.
        
        Returns:
            Statistics dictionary
        """
        docs = self.list_uploaded_documents()
        
        return {
            "total_documents": len(docs),
            "total_size_mb": round(sum(d["size_mb"] for d in docs), 2),
            "by_type": self._count_by_type(docs),
            "processed": sum(1 for d in docs if d["processed"]),
            "pending": sum(1 for d in docs if not d["processed"])
        }

    def _count_by_type(self, docs: List[Dict]) -> Dict[str, int]:
        """Count documents by type."""
        counts = {}
        for doc in docs:
            doc_type = doc["type"]
            counts[doc_type] = counts.get(doc_type, 0) + 1
        return counts

    def delete_document(self, filename: str) -> Tuple[bool, str]:
        """
        Delete an uploaded document and its processing metadata.
        
        Args:
            filename: Name of file to delete
            
        Returns:
            Tuple: (success: bool, message: str)
        """
        try:
            file_path = self.storage_dir / filename
            
            if not file_path.exists():
                return False, f"❌ File not found: {filename}"
            
            file_path.unlink()
            
            # Also delete processed file if it exists
            processed_file = self.training_dir / f"{Path(filename).stem}_extracted.txt"
            if processed_file.exists():
                processed_file.unlink()
                logger.info(f"Deleted processed file: {processed_file.name}")
            
            # Remove from processing metadata
            if filename in self._processing_metadata:
                del self._processing_metadata[filename]
                self._save_processing_metadata()
                logger.info(f"Removed from processing metadata: {filename}")
            
            logger.info(f"Deleted document: {filename}")
            return True, f"✅ Deleted '{filename}'"
            
        except Exception as e:
            error_msg = f"Delete failed: {str(e)}"
            logger.error(error_msg)
            return False, f"❌ {error_msg}"

    def get_pending_documents(self) -> List[Dict[str, Any]]:
        """Get documents that haven't been processed yet."""
        docs = self.list_uploaded_documents()
        return [d for d in docs if not d["processed"]]

    def mark_processed(self, filename: str, processed_path: str) -> bool:
        """
        Mark a document as processed and persist the state to JSON.
        
        Args:
            filename: Original filename
            processed_path: Path to processed output file
            
        Returns:
            bool: Success status
        """
        try:
            # Update in-memory state
            self._processing_metadata[filename] = {
                'processed': True,
                'processed_path': processed_path,
                'processed_date': datetime.now().isoformat()
            }
            
            # Persist to disk
            self._save_processing_metadata()
            
            logger.info(f"Marked as processed: {filename} → {processed_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark processed: {str(e)}")
            return False

    def _load_processing_metadata(self) -> Dict[str, Any]:
        """
        Load processing metadata from disk (JSON file).
        
        Returns:
            Dict: Processing metadata (filename -> processing status)
        """
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load processing metadata: {str(e)}")
        
        return {}

    def _save_processing_metadata(self) -> bool:
        """
        Save processing metadata to disk (JSON file).
        
        Returns:
            bool: Success status
        """
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._processing_metadata, f, indent=2)
            logger.debug(f"Saved processing metadata ({len(self._processing_metadata)} entries)")
            return True
        except Exception as e:
            logger.error(f"Failed to save processing metadata: {str(e)}")
            return False


# Singleton instance
_document_manager = None


def get_document_manager() -> DocumentManager:
    """
    Get or create singleton DocumentManager instance.
    
    Returns:
        DocumentManager: Singleton instance
    """
    global _document_manager
    if _document_manager is None:
        _document_manager = DocumentManager()
    return _document_manager
