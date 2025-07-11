# file_butler/core/extractor.py
import os
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import logging

from .models import FileInfo, FileType

logger = logging.getLogger(__name__)

class BaseFeatureExtractor(ABC):
    """Base class for feature extractors"""
    
    @abstractmethod
    def extract(self, file_info: FileInfo) -> Dict[str, Any]:
        """Extract features from a file"""
        pass
    
    @abstractmethod
    def applicable_types(self) -> List[FileType]:
        """Return file types this extractor can handle"""
        pass

class MetadataExtractor(BaseFeatureExtractor):
    """Extracts basic file system metadata"""
    
    def extract(self, file_info: FileInfo) -> Dict[str, Any]:
        """Extract metadata features"""
        name_lower = file_info.name.lower()
        
        return {
            'extension': file_info.extension,
            'size_mb': file_info.size_mb,
            'age_days': file_info.age_days,
            'filename_length': len(file_info.name),
            'has_spaces': ' ' in file_info.name,
            'has_numbers': any(c.isdigit() for c in file_info.name),
            'is_uppercase': file_info.name.isupper(),
            'is_lowercase': file_info.name.islower(),
            'word_count': len(file_info.name.split()),
            
            # Common filename patterns
            'likely_download': any(word in name_lower for word in ['download', 'temp', 'tmp']),
            'likely_screenshot': any(word in name_lower for word in ['screenshot', 'screen shot', 'capture']),
            'likely_backup': any(word in name_lower for word in ['backup', 'copy', 'bak']),
            'has_date_pattern': self._has_date_pattern(file_info.name),
            'has_version_pattern': self._has_version_pattern(file_info.name),
        }
    
    def applicable_types(self) -> List[FileType]:
        return list(FileType)  # Applies to all file types
    
    def _has_date_pattern(self, filename: str) -> bool:
        """Check if filename contains date-like patterns"""
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}\d{2}\d{2}',    # YYYYMMDD
        ]
        return any(re.search(pattern, filename) for pattern in date_patterns)
    
    def _has_version_pattern(self, filename: str) -> bool:
        """Check if filename contains version-like patterns"""
        import re
        version_patterns = [
            r'v\d+',           # v1, v2, etc.
            r'_\d+$',          # ending with _1, _2, etc.
            r'\(\d+\)',        # (1), (2), etc.
        ]
        return any(re.search(pattern, filename) for pattern in version_patterns)

class ContentHashExtractor(BaseFeatureExtractor):
    """Extracts content-based hash for duplicate detection"""
    
    def __init__(self, hash_algorithm='sha256', chunk_size=4096):
        self.hash_algorithm = hash_algorithm
        self.chunk_size = chunk_size
    
    def extract(self, file_info: FileInfo) -> Dict[str, Any]:
        """Extract content hash"""
        try:
            hasher = hashlib.new(self.hash_algorithm)
            
            with open(file_info.path, 'rb') as f:
                while chunk := f.read(self.chunk_size):
                    hasher.update(chunk)
            
            return {
                'content_hash': hasher.hexdigest(),
                'hash_algorithm': self.hash_algorithm
            }
            
        except (IOError, OSError) as e:
            logger.warning(f"Cannot hash file {file_info.path}: {e}")
            return {'content_hash': None, 'hash_algorithm': self.hash_algorithm}
    
    def applicable_types(self) -> List[FileType]:
        return list(FileType)  # Applies to all file types

class TextContentExtractor(BaseFeatureExtractor):
    """Extracts features from text-based files"""
    
    def __init__(self, max_chars=5000, encoding='utf-8'):
        self.max_chars = max_chars
        self.encoding = encoding
    
    def extract(self, file_info: FileInfo) -> Dict[str, Any]:
        """Extract text content features"""
        try:
            content = self._read_text_content(file_info.path)
            
            if not content:
                return {'has_content': False}
            
            words = content.lower().split()
            
            return {
                'has_content': True,
                'char_count': len(content),
                'word_count': len(words),
                'line_count': content.count('\n'),
                'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
                'unique_words': len(set(words)),
                'has_urls': 'http' in content.lower(),
                'has_emails': '@' in content and '.' in content,
                'content_preview': content[:200] + '...' if len(content) > 200 else content,
                
                # Simple keyword extraction (top 10 most common words)
                'top_keywords': self._extract_keywords(words)
            }
            
        except Exception as e:
            logger.warning(f"Cannot extract text from {file_info.path}: {e}")
            return {'has_content': False, 'error': str(e)}
    
    def applicable_types(self) -> List[FileType]:
        return [FileType.DOCUMENT, FileType.CODE]
    
    def _read_text_content(self, file_path: str) -> str:
        """Read text content from file"""
        try:
            with open(file_path, 'r', encoding=self.encoding, errors='ignore') as f:
                return f.read(self.max_chars)
        except Exception:
            return ""
    
    def _extract_keywords(self, words: List[str], top_k=10) -> List[str]:
        """Extract top keywords (simple frequency-based)"""
        from collections import Counter
        
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        word_counts = Counter(filtered_words)
        return [word for word, _ in word_counts.most_common(top_k)]

class ImageMetadataExtractor(BaseFeatureExtractor):
    """Extracts metadata from image files"""
    
    def extract(self, file_info: FileInfo) -> Dict[str, Any]:
        """Extract image metadata using basic techniques"""
        features = {}
        
        try:
            # Try to get basic image info
            from PIL import Image
            
            with Image.open(file_info.path) as img:
                features.update({
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'aspect_ratio': img.width / img.height if img.height > 0 else 0,
                    'total_pixels': img.width * img.height,
                    'is_landscape': img.width > img.height,
                    'is_portrait': img.height > img.width,
                    'is_square': abs(img.width - img.height) < 10,
                })
                
                # Try to extract EXIF data if available
                exif_data = getattr(img, '_getexif', lambda: None)()
                if exif_data:
                    features['has_exif'] = True
                    # Add basic EXIF info parsing here if needed
                else:
                    features['has_exif'] = False
                    
        except Exception as e:
            logger.warning(f"Cannot extract image metadata from {file_info.path}: {e}")
            features = {'extraction_error': str(e)}
        
        return features
    
    def applicable_types(self) -> List[FileType]:
        return [FileType.IMAGE]

class FeatureExtractorManager:
    """Manages multiple feature extractors"""
    
    def __init__(self):
        self.extractors = []
        self._register_default_extractors()
    
    def _register_default_extractors(self):
        """Register default extractors"""
        self.add_extractor(MetadataExtractor())
        self.add_extractor(ContentHashExtractor())
        
        # Add text extractor if possible
        self.add_extractor(TextContentExtractor())
        
        # Add image extractor if PIL is available
        try:
            self.add_extractor(ImageMetadataExtractor())
        except ImportError:
            logger.info("PIL not available, skipping image metadata extraction")
    
    def add_extractor(self, extractor: BaseFeatureExtractor):
        """Add a feature extractor"""
        self.extractors.append(extractor)
    
    def extract_features(self, file_info: FileInfo) -> Dict[str, Any]:
        """Extract all applicable features for a file"""
        all_features = {}
        
        for extractor in self.extractors:
            if file_info.file_type in extractor.applicable_types():
                try:
                    features = extractor.extract(file_info)
                    if features:
                        # Prefix features with extractor class name to avoid conflicts
                        extractor_name = extractor.__class__.__name__.replace('Extractor', '').lower()
                        all_features[extractor_name] = features
                        
                except Exception as e:
                    logger.error(f"Error in {extractor.__class__.__name__}: {e}")
        
        return all_features
    
    def extract_batch(self, files: List[FileInfo]) -> List[FileInfo]:
        """Extract features for multiple files"""
        for file_info in files:
            file_info.features = self.extract_features(file_info)
        
        return files