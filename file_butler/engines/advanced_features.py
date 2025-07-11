# file_butler/engines/advanced_features.py
"""Advanced feature extractors for ML and LLM engines"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from ..core.models import FileInfo, FileType

logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor:
    """Extract advanced features for ML/LLM processing"""
    
    def __init__(self):
        self.embeddings_cache = {}
    
    def extract_document_embeddings(self, files: List[FileInfo]) -> Optional[np.ndarray]:
        """Extract document embeddings using sentence transformers"""
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
            return None
        
        # Filter to text documents
        text_files = [f for f in files if f.file_type in [FileType.DOCUMENT, FileType.CODE]]
        
        if not text_files:
            return None
        
        # Load model (lightweight model for speed)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prepare text content
        texts = []
        for file_info in text_files:
            # Combine filename and any available content
            text_parts = [file_info.name]
            
            if hasattr(file_info, 'features') and file_info.features:
                content_features = file_info.features.get('textcontent', {})
                if 'content_preview' in content_features:
                    text_parts.append(content_features['content_preview'])
            
            texts.append(' '.join(text_parts))
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False)
        
        logger.info(f"Generated embeddings for {len(text_files)} documents")
        return embeddings
    
    def extract_image_features(self, files: List[FileInfo]) -> Optional[np.ndarray]:
        """Extract image features using deep learning"""
        
        # This would require a more complex setup with PyTorch/TensorFlow
        # For now, return placeholder that uses metadata
        
        image_files = [f for f in files if f.file_type == FileType.IMAGE]
        
        if not image_files:
            return None
        
        # Use available image metadata from our existing extractor
        features = []
        for file_info in image_files:
            if hasattr(file_info, 'features') and file_info.features:
                img_features = file_info.features.get('imagemetadata', {})
                
                # Create feature vector from metadata
                feature_vector = [
                    img_features.get('width', 0),
                    img_features.get('height', 0),
                    img_features.get('aspect_ratio', 1.0),
                    img_features.get('total_pixels', 0),
                    int(img_features.get('is_landscape', False)),
                    int(img_features.get('is_portrait', False)),
                    int(img_features.get('is_square', False)),
                    int(img_features.get('has_exif', False))
                ]
                
                features.append(feature_vector)
            else:
                # Default feature vector
                features.append([0] * 8)
        
        return np.array(features) if features else None
    
    def extract_semantic_similarity(self, files: List[FileInfo]) -> np.ndarray:
        """Extract semantic similarity matrix between files"""
        
        n_files = len(files)
        similarity_matrix = np.eye(n_files)  # Start with identity matrix
        
        # Compute pairwise similarities based on various factors
        for i in range(n_files):
            for j in range(i + 1, n_files):
                similarity = self._compute_file_similarity(files[i], files[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _compute_file_similarity(self, file1: FileInfo, file2: FileInfo) -> float:
        """Compute similarity between two files"""
        
        similarity_scores = []
        
        # Type similarity
        if file1.file_type == file2.file_type:
            similarity_scores.append(0.3)
        
        # Extension similarity
        if file1.extension == file2.extension:
            similarity_scores.append(0.2)
        
        # Name similarity (Jaccard similarity of words)
        words1 = set(file1.name.lower().split())
        words2 = set(file2.name.lower().split())
        
        if words1 and words2:
            jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
            similarity_scores.append(jaccard * 0.3)
        
        # Size similarity (normalized)
        if file1.size > 0 and file2.size > 0:
            size_ratio = min(file1.size, file2.size) / max(file1.size, file2.size)
            similarity_scores.append(size_ratio * 0.1)
        
        # Date similarity
        time_diff = abs(file1.modified_at - file2.modified_at)
        date_similarity = max(0, 1 - time_diff / (30 * 24 * 3600))  # 30 days max
        similarity_scores.append(date_similarity * 0.1)
        
        return sum(similarity_scores)

class ImageHashExtractor:
    """Extract perceptual hashes for image similarity"""
    
    def __init__(self):
        self.hash_cache = {}
    
    def extract_perceptual_hashes(self, files: List[FileInfo]) -> Dict[str, str]:
        """Extract perceptual hashes for images"""
        
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            logger.warning("imagehash/PIL not available. Install with: pip install imagehash Pillow")
            return {}
        
        hashes = {}
        image_files = [f for f in files if f.file_type == FileType.IMAGE]
        
        for file_info in image_files:
            try:
                if file_info.path in self.hash_cache:
                    hashes[file_info.path] = self.hash_cache[file_info.path]
                    continue
                
                with Image.open(file_info.path) as img:
                    # Use average hash (fast and effective)
                    hash_value = str(imagehash.average_hash(img))
                    hashes[file_info.path] = hash_value
                    self.hash_cache[file_info.path] = hash_value
                    
            except Exception as e:
                logger.warning(f"Cannot hash image {file_info.name}: {e}")
        
        return hashes
    
    def find_similar_images(self, files: List[FileInfo], threshold: int = 5) -> List[List[FileInfo]]:
        """Find groups of similar images based on perceptual hashes"""
        
        hashes = self.extract_perceptual_hashes(files)
        
        if not hashes:
            return []
        
        try:
            import imagehash
        except ImportError:
            return []
        
        # Group similar hashes
        similar_groups = []
        processed = set()
        
        hash_to_file = {hash_val: path for path, hash_val in hashes.items()}
        file_lookup = {f.path: f for f in files}
        
        for path1, hash1_str in hashes.items():
            if path1 in processed:
                continue
            
            current_group = [file_lookup[path1]]
            processed.add(path1)
            
            hash1 = imagehash.hex_to_hash(hash1_str)
            
            for path2, hash2_str in hashes.items():
                if path2 in processed:
                    continue
                
                hash2 = imagehash.hex_to_hash(hash2_str)
                
                # Calculate Hamming distance
                if hash1 - hash2 <= threshold:
                    current_group.append(file_lookup[path2])
                    processed.add(path2)
            
            if len(current_group) > 1:
                similar_groups.append(current_group)
        
        return similar_groups