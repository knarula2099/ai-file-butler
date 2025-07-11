# file_butler/detectors/duplicates.py
import hashlib
from collections import defaultdict
from typing import List, Dict, Set
import logging

from ..core.models import FileInfo, Action, ActionType

logger = logging.getLogger(__name__)

class DuplicateDetector:
    """Detects exact and near-duplicate files"""
    
    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
        self.hash_cache = {}
    
    def find_exact_duplicates(self, files: List[FileInfo]) -> List[List[FileInfo]]:
        """
        Find exact duplicate files using size + hash comparison
        
        Returns:
            List of duplicate groups, where each group contains identical files
        """
        
        # Step 1: Group by size (optimization)
        size_groups = defaultdict(list)
        for file_info in files:
            size_groups[file_info.size].append(file_info)
        
        duplicate_groups = []
        
        # Step 2: For each size group with multiple files, compare hashes
        for size, file_group in size_groups.items():
            if len(file_group) < 2:
                continue  # No duplicates possible with single file
            
            if size == 0:
                # All empty files are considered duplicates
                duplicate_groups.append(file_group)
                continue
            
            # Hash files in this size group
            hash_groups = defaultdict(list)
            
            for file_info in file_group:
                file_hash = self._get_file_hash(file_info.path)
                if file_hash:
                    hash_groups[file_hash].append(file_info)
            
            # Collect groups with actual hash duplicates
            for hash_val, hash_group in hash_groups.items():
                if len(hash_group) > 1:
                    duplicate_groups.append(hash_group)
        
        logger.info(f"Found {len(duplicate_groups)} groups of exact duplicates")
        return duplicate_groups
    
    def suggest_deletions(self, duplicate_groups: List[List[FileInfo]]) -> List[Action]:
        """
        Suggest which duplicate files to delete
        
        Strategy: Keep the most recently modified file in each group
        """
        
        actions = []
        
        for group in duplicate_groups:
            if len(group) < 2:
                continue
            
            # Sort by modification time (newest first)
            sorted_group = sorted(group, key=lambda x: x.modified_at, reverse=True)
            
            # Keep the first (newest) file, suggest deleting the rest
            keep_file = sorted_group[0]
            delete_files = sorted_group[1:]
            
            for file_info in delete_files:
                actions.append(Action(
                    action_type=ActionType.DELETE,
                    source=file_info.path,
                    reason=f"Duplicate of {keep_file.name} (keeping newer version from {self._format_date(keep_file.modified_at)})",
                    confidence=0.95  # High confidence for exact duplicates
                ))
        
        return actions
    
    def get_duplicate_statistics(self, duplicate_groups: List[List[FileInfo]]) -> Dict:
        """Get statistics about duplicates"""
        
        total_files = sum(len(group) for group in duplicate_groups)
        total_duplicates = total_files - len(duplicate_groups)  # Subtract one "original" per group
        
        total_size = 0
        wasted_space = 0
        
        for group in duplicate_groups:
            if group:
                file_size = group[0].size
                total_size += file_size * len(group)
                wasted_space += file_size * (len(group) - 1)  # All but one file is "wasted"
        
        return {
            'duplicate_groups': len(duplicate_groups),
            'total_files_in_groups': total_files,
            'total_duplicates': total_duplicates,
            'total_size_mb': total_size / (1024 * 1024),
            'wasted_space_mb': wasted_space / (1024 * 1024),
            'space_savings_percent': (wasted_space / total_size * 100) if total_size > 0 else 0
        }
    
    def _get_file_hash(self, filepath: str) -> str:
        """
        Get SHA-256 hash of file content with caching
        
        Args:
            filepath: Path to file to hash
            
        Returns:
            Hex digest of SHA-256 hash, or None if file cannot be read
        """
        
        # Check cache first
        if filepath in self.hash_cache:
            return self.hash_cache[filepath]
        
        try:
            hasher = hashlib.sha256()
            
            with open(filepath, 'rb') as f:
                # Read file in chunks to handle large files efficiently
                while chunk := f.read(self.chunk_size):
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            self.hash_cache[filepath] = file_hash
            return file_hash
            
        except (IOError, OSError, PermissionError) as e:
            logger.warning(f"Cannot hash file {filepath}: {e}")
            return None
    
    def _format_date(self, timestamp: float) -> str:
        """Format timestamp as readable date"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
    
    def clear_cache(self):
        """Clear the hash cache"""
        self.hash_cache.clear()


class NearDuplicateDetector:
    """Detects files that are similar but not identical"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def find_similar_text_files(self, files: List[FileInfo]) -> List[List[FileInfo]]:
        """
        Find text files with similar content using simple techniques
        
        This is a simplified version - for production, you'd want to use
        more sophisticated techniques like MinHash + LSH
        """
        
        # Filter to text files only
        text_files = [f for f in files if f.file_type.value in ['document', 'code']]
        
        if len(text_files) < 2:
            return []
        
        similar_groups = []
        processed = set()
        
        for i, file1 in enumerate(text_files):
            if file1.path in processed:
                continue
                
            current_group = [file1]
            processed.add(file1.path)
            
            # Compare with remaining files
            for j, file2 in enumerate(text_files[i+1:], i+1):
                if file2.path in processed:
                    continue
                
                similarity = self._calculate_text_similarity(file1, file2)
                
                if similarity >= self.similarity_threshold:
                    current_group.append(file2)
                    processed.add(file2.path)
            
            # Only add groups with multiple files
            if len(current_group) > 1:
                similar_groups.append(current_group)
        
        return similar_groups
    
    def _calculate_text_similarity(self, file1: FileInfo, file2: FileInfo) -> float:
        """
        Calculate similarity between two text files
        
        Simple implementation using filename and size similarity
        For production, you'd want to compare actual content
        """
        
        # Name similarity (Jaccard similarity of words)
        words1 = set(file1.name.lower().split())
        words2 = set(file2.name.lower().split())
        
        if not words1 and not words2:
            name_similarity = 1.0
        elif not words1 or not words2:
            name_similarity = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            name_similarity = intersection / union
        
        # Size similarity
        size1, size2 = file1.size, file2.size
        if size1 == 0 and size2 == 0:
            size_similarity = 1.0
        elif size1 == 0 or size2 == 0:
            size_similarity = 0.0
        else:
            size_ratio = min(size1, size2) / max(size1, size2)
            size_similarity = size_ratio
        
        # Combined similarity (weighted average)
        return 0.7 * name_similarity + 0.3 * size_similarity
    
    def suggest_review_actions(self, similar_groups: List[List[FileInfo]]) -> List[Action]:
        """Suggest actions for similar files (manual review recommended)"""
        
        actions = []
        
        for group in similar_groups:
            if len(group) < 2:
                continue
            
            # Create a review action for the group
            file_list = ", ".join([f.name for f in group])
            actions.append(Action(
                action_type=ActionType.MOVE,
                source=group[0].path,  # Representative file
                destination="_Needs_Review/Similar_Files",
                reason=f"Similar to: {file_list}",
                confidence=0.6,  # Lower confidence for near-duplicates
                metadata={'similar_group': [f.path for f in group]}
            ))
        
        return actions