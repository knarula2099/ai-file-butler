# file_butler/core/scanner.py
import os
import time
from pathlib import Path
from typing import List, Optional, Callable
import logging

from .models import FileInfo, ScanResult

logger = logging.getLogger(__name__)

class FileScanner:
    """Scans directories and builds file inventory"""
    
    def __init__(self, 
                 include_hidden: bool = False,
                 max_depth: Optional[int] = None,
                 size_limit_mb: Optional[float] = None,
                 progress_callback: Optional[Callable[[int, str], None]] = None):
        """
        Args:
            include_hidden: Whether to include hidden files/directories
            max_depth: Maximum directory depth to scan (None = unlimited)
            size_limit_mb: Skip files larger than this (None = no limit)
            progress_callback: Function called with (count, current_file) during scan
        """
        self.include_hidden = include_hidden
        self.max_depth = max_depth
        self.size_limit_mb = size_limit_mb
        self.progress_callback = progress_callback
        
    def scan_directory(self, directory: str) -> ScanResult:
        """
        Scan directory and return structured results
        
        Args:
            directory: Path to directory to scan
            
        Returns:
            ScanResult with files and metadata
        """
        start_time = time.time()
        directory = Path(directory).resolve()
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        files = []
        errors = []
        
        logger.info(f"Starting scan of: {directory}")
        
        try:
            for file_path in self._walk_directory(directory):
                try:
                    file_info = self._create_file_info(file_path)
                    if file_info:
                        files.append(file_info)
                        
                        if self.progress_callback:
                            self.progress_callback(len(files), str(file_path))
                            
                except (OSError, IOError, PermissionError) as e:
                    error_msg = f"Cannot access {file_path}: {e}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                    
        except Exception as e:
            error_msg = f"Scan failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        scan_time = time.time() - start_time
        
        logger.info(f"Scan completed: {len(files)} files found in {scan_time:.2f}s")
        
        return ScanResult(
            files=files,
            source_directory=str(directory),
            scan_time=scan_time,
            errors=errors
        )
    
    def _walk_directory(self, directory: Path, current_depth: int = 0):
        """Generator that walks directory tree with depth control"""
        
        if self.max_depth is not None and current_depth > self.max_depth:
            return
            
        try:
            for item in directory.iterdir():
                # Skip hidden files/directories if not included
                if not self.include_hidden and item.name.startswith('.'):
                    continue
                
                if item.is_file():
                    yield item
                elif item.is_dir():
                    # Recursively scan subdirectories
                    yield from self._walk_directory(item, current_depth + 1)
                    
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot access directory {directory}: {e}")
    
    def _create_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """Create FileInfo object from file path"""
        
        try:
            stat = file_path.stat()
            
            # Apply size filter
            if self.size_limit_mb:
                size_mb = stat.st_size / (1024 * 1024)
                if size_mb > self.size_limit_mb:
                    logger.debug(f"Skipping large file: {file_path} ({size_mb:.1f}MB)")
                    return None
            
            return FileInfo(
                path=str(file_path),
                name=file_path.name,
                extension=file_path.suffix.lower(),
                size=stat.st_size,
                created_at=stat.st_ctime,
                modified_at=stat.st_mtime,
                accessed_at=stat.st_atime
            )
            
        except (OSError, IOError) as e:
            logger.warning(f"Cannot get stats for {file_path}: {e}")
            return None

class FilteredScanner(FileScanner):
    """Scanner with additional filtering capabilities"""
    
    def __init__(self, 
                 extensions: Optional[List[str]] = None,
                 exclude_extensions: Optional[List[str]] = None,
                 min_age_days: Optional[int] = None,
                 max_age_days: Optional[int] = None,
                 **kwargs):
        """
        Args:
            extensions: Only include files with these extensions
            exclude_extensions: Exclude files with these extensions
            min_age_days: Only include files older than this
            max_age_days: Only include files newer than this
        """
        super().__init__(**kwargs)
        self.extensions = {ext.lower() for ext in (extensions or [])}
        self.exclude_extensions = {ext.lower() for ext in (exclude_extensions or [])}
        self.min_age_days = min_age_days
        self.max_age_days = max_age_days
    
    def _create_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """Create FileInfo with additional filtering"""
        
        # Apply base filtering first
        file_info = super()._create_file_info(file_path)
        if not file_info:
            return None
        
        # Extension filtering
        if self.extensions and file_info.extension not in self.extensions:
            return None
        
        if self.exclude_extensions and file_info.extension in self.exclude_extensions:
            return None
        
        # Age filtering
        if self.min_age_days and file_info.age_days < self.min_age_days:
            return None
        
        if self.max_age_days and file_info.age_days > self.max_age_days:
            return None
        
        return file_info