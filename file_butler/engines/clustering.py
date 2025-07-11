# file_butler/engines/clustering.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from collections import Counter
from dataclasses import dataclass

from ..core.models import FileInfo, Action, ActionType, OrganizationPlan, FileType

logger = logging.getLogger(__name__)

@dataclass
class ClusteringConfig:
    """Configuration for clustering engine"""
    algorithm: str = "dbscan"  # "dbscan", "kmeans", "hierarchical"
    min_samples: int = 2
    eps: float = 0.5
    n_clusters: Optional[int] = None  # For k-means
    max_features: int = 1000
    use_tfidf: bool = True
    min_cluster_size: int = 2
    feature_types: List[str] = None  # Which features to use
    
    def __post_init__(self):
        if self.feature_types is None:
            self.feature_types = ['text', 'metadata', 'name_similarity']

class MLClusteringEngine:
    """Machine learning-based file clustering and organization"""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Check if required ML libraries are available"""
        try:
            import sklearn
            import pandas
            import numpy
        except ImportError as e:
            raise ImportError(f"Required ML libraries not installed: {e}. Run: pip install scikit-learn pandas numpy")
    
    def organize(self, files: List[FileInfo]) -> OrganizationPlan:
        """Generate organization plan using ML clustering"""
        
        logger.info(f"Starting ML clustering for {len(files)} files")
        
        if len(files) < self.config.min_cluster_size:
            logger.warning("Too few files for meaningful clustering, falling back to basic organization")
            return self._fallback_organization(files)
        
        # Step 1: Extract features for clustering
        feature_matrix, feature_names = self._extract_clustering_features(files)
        
        if feature_matrix.shape[1] == 0:
            logger.warning("No features extracted, falling back to basic organization")
            return self._fallback_organization(files)
        
        # Step 2: Perform clustering
        cluster_labels = self._perform_clustering(feature_matrix)
        
        # Step 3: Analyze clusters and generate folder names
        cluster_analysis = self._analyze_clusters(files, cluster_labels, feature_names)
        
        # Step 4: Generate organization plan
        actions = self._generate_clustering_actions(files, cluster_labels, cluster_analysis)
        
        logger.info(f"Generated {len(actions)} actions using ML clustering")
        
        return OrganizationPlan(
            actions=actions,
            source_directory="",  # Will be set by caller
            target_directory="organized"
        )
    
    def _extract_clustering_features(self, files: List[FileInfo]) -> Tuple[np.ndarray, List[str]]:
        """Extract features suitable for clustering"""
        
        features_list = []
        feature_names = []
        
        # Text-based features (TF-IDF)
        if 'text' in self.config.feature_types:
            text_features, text_feature_names = self._extract_text_features(files)
            if text_features.shape[1] > 0:
                features_list.append(text_features)
                feature_names.extend(text_feature_names)
        
        # Metadata features
        if 'metadata' in self.config.feature_types:
            metadata_features, metadata_names = self._extract_metadata_features(files)
            if metadata_features.shape[1] > 0:
                features_list.append(metadata_features)
                feature_names.extend(metadata_names)
        
        # Name similarity features
        if 'name_similarity' in self.config.feature_types:
            name_features, name_feature_names = self._extract_name_features(files)
            if name_features.shape[1] > 0:
                features_list.append(name_features)
                feature_names.extend(name_feature_names)
        
        if not features_list:
            logger.warning("No features could be extracted")
            return np.array([]).reshape(len(files), 0), []
        
        # Combine all features
        combined_features = np.hstack(features_list)
        
        logger.info(f"Extracted {combined_features.shape[1]} features for {combined_features.shape[0]} files")
        
        return combined_features, feature_names
    
    def _extract_text_features(self, files: List[FileInfo]) -> Tuple[np.ndarray, List[str]]:
        """Extract TF-IDF features from file names and content"""
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            logger.warning("scikit-learn not available for text features")
            return np.array([]).reshape(len(files), 0), []
        
        # Collect text data
        text_data = []
        for file_info in files:
            # Combine filename, extension, and any available text content
            text_parts = [
                file_info.name,
                file_info.extension,
                file_info.file_type.value
            ]
            
            # Add text content if available in features
            if hasattr(file_info, 'features') and file_info.features:
                text_content = file_info.features.get('textcontent', {})
                if 'content_preview' in text_content:
                    text_parts.append(text_content['content_preview'])
                if 'top_keywords' in text_content:
                    text_parts.extend(text_content['top_keywords'])
            
            text_data.append(' '.join(text_parts))
        
        # Create TF-IDF vectors
        try:
            vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            tfidf_matrix = vectorizer.fit_transform(text_data)
            feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
            
            return tfidf_matrix.toarray(), feature_names
            
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
            return np.array([]).reshape(len(files), 0), []
    
    def _extract_metadata_features(self, files: List[FileInfo]) -> Tuple[np.ndarray, List[str]]:
        """Extract numerical features from file metadata"""
        
        features = []
        feature_names = []
        
        for file_info in files:
            file_features = []
            
            # Basic metadata
            file_features.extend([
                np.log1p(file_info.size),  # Log transform for size
                file_info.age_days,
                len(file_info.name),
                file_info.name.count('_'),
                file_info.name.count('-'),
                file_info.name.count(' '),
                int(file_info.name.isupper()),
                int(file_info.name.islower()),
                int(any(c.isdigit() for c in file_info.name))
            ])
            
            # File type one-hot encoding
            for file_type in FileType:
                file_features.append(int(file_info.file_type == file_type))
            
            # Extension features
            common_extensions = ['.txt', '.pdf', '.jpg', '.png', '.mp4', '.zip', '.docx']
            for ext in common_extensions:
                file_features.append(int(file_info.extension.lower() == ext))
            
            features.append(file_features)
        
        if not features:
            return np.array([]).reshape(len(files), 0), []
        
        feature_names = [
            'log_size', 'age_days', 'name_length', 'underscore_count',
            'dash_count', 'space_count', 'is_upper', 'is_lower', 'has_digits'
        ]
        feature_names.extend([f"type_{ft.value}" for ft in FileType])
        feature_names.extend([f"ext_{ext}" for ext in common_extensions])
        
        return np.array(features), feature_names
    
    def _extract_name_features(self, files: List[FileInfo]) -> Tuple[np.ndarray, List[str]]:
        """Extract features based on filename similarity patterns"""
        
        # Extract common patterns and prefixes
        all_names = [file_info.name for file_info in files]
        
        # Find common prefixes/suffixes
        common_patterns = self._find_common_patterns(all_names)
        
        features = []
        for file_info in files:
            file_features = []
            
            # Pattern matching
            for pattern in common_patterns:
                file_features.append(int(pattern in file_info.name.lower()))
            
            # Date patterns
            import re
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{8}',              # YYYYMMDD
                r'\d{4}_\d{2}_\d{2}'   # YYYY_MM_DD
            ]
            
            for pattern in date_patterns:
                file_features.append(int(bool(re.search(pattern, file_info.name))))
            
            features.append(file_features)
        
        if not features:
            return np.array([]).reshape(len(files), 0), []
        
        feature_names = [f"pattern_{i}" for i in range(len(common_patterns))]
        feature_names.extend(['date_dash', 'date_slash', 'date_compact', 'date_underscore'])
        
        return np.array(features), feature_names
    
    def _find_common_patterns(self, names: List[str], min_frequency: int = 2) -> List[str]:
        """Find common patterns in filenames"""
        
        # Extract words and common substrings
        all_words = []
        for name in names:
            # Split on common delimiters
            import re
            words = re.split(r'[_\-\s\.]+', name.lower())
            all_words.extend([w for w in words if len(w) > 2])
        
        # Find frequent patterns
        word_counts = Counter(all_words)
        common_patterns = [word for word, count in word_counts.items() 
                          if count >= min_frequency and len(word) > 2]
        
        return common_patterns[:20]  # Limit to top 20 patterns
    
    def _perform_clustering(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Perform clustering on feature matrix"""
        
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        if self.config.algorithm == "dbscan":
            return self._dbscan_clustering(scaled_features)
        elif self.config.algorithm == "kmeans":
            return self._kmeans_clustering(scaled_features)
        elif self.config.algorithm == "hierarchical":
            return self._hierarchical_clustering(scaled_features)
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.config.algorithm}")
    
    def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering"""
        
        from sklearn.cluster import DBSCAN
        
        clustering = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples,
            metric='cosine'  # Good for text features
        )
        
        labels = clustering.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"DBSCAN found {n_clusters} clusters with {n_noise} noise points")
        
        return labels
    
    def _kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform K-means clustering"""
        
        from sklearn.cluster import KMeans
        
        # Determine number of clusters if not specified
        n_clusters = self.config.n_clusters
        if n_clusters is None:
            # Use elbow method or simple heuristic
            n_clusters = min(10, max(2, len(features) // 5))
        
        clustering = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        labels = clustering.fit_predict(features)
        
        logger.info(f"K-means created {n_clusters} clusters")
        
        return labels
    
    def _hierarchical_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering"""
        
        from sklearn.cluster import AgglomerativeClustering
        
        # Determine number of clusters
        n_clusters = self.config.n_clusters
        if n_clusters is None:
            n_clusters = min(8, max(2, len(features) // 4))
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        
        labels = clustering.fit_predict(features)
        
        logger.info(f"Hierarchical clustering created {n_clusters} clusters")
        
        return labels
    
    def _analyze_clusters(self, files: List[FileInfo], labels: np.ndarray, 
                         feature_names: List[str]) -> Dict[int, Dict[str, Any]]:
        """Analyze clusters to understand their characteristics"""
        
        cluster_analysis = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise cluster in DBSCAN
                continue
            
            cluster_files = [files[i] for i, l in enumerate(labels) if l == label]
            
            if len(cluster_files) < self.config.min_cluster_size:
                continue
            
            analysis = {
                'files': cluster_files,
                'size': len(cluster_files),
                'name': self._generate_cluster_name(cluster_files),
                'dominant_type': self._get_dominant_file_type(cluster_files),
                'common_patterns': self._find_cluster_patterns(cluster_files),
                'avg_size': np.mean([f.size for f in cluster_files]),
                'date_range': self._get_date_range(cluster_files)
            }
            
            cluster_analysis[label] = analysis
        
        return cluster_analysis
    
    def _generate_cluster_name(self, cluster_files: List[FileInfo]) -> str:
        """Generate a descriptive name for a cluster"""
        
        # Analyze file names for common patterns
        names = [f.name.lower() for f in cluster_files]
        
        # Find common words
        all_words = []
        for name in names:
            import re
            words = re.split(r'[_\-\s\.]+', name)
            all_words.extend([w for w in words if len(w) > 2])
        
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(3) 
                       if count >= 2]
        
        # Check file types
        file_types = [f.file_type for f in cluster_files]
        type_counts = Counter(file_types)
        dominant_type = type_counts.most_common(1)[0][0]
        
        # Generate name based on patterns
        if common_words:
            # Use most common meaningful words
            meaningful_words = [w for w in common_words 
                              if w not in ['file', 'document', 'image', 'the', 'and', 'or']]
            if meaningful_words:
                name_base = '_'.join(meaningful_words[:2]).title()
                return f"{name_base}_{dominant_type.value.title()}"
        
        # Check for date patterns
        import re
        date_pattern_found = False
        for name in names:
            if re.search(r'\d{4}', name):  # Contains year
                date_pattern_found = True
                break
        
        if date_pattern_found:
            return f"Dated_{dominant_type.value.title()}"
        
        # Check for common prefixes
        if len(names) > 1:
            import os
            common_prefix = os.path.commonprefix(names)
            if len(common_prefix) > 3:
                clean_prefix = re.sub(r'[^a-zA-Z0-9]', '', common_prefix)
                if clean_prefix:
                    return f"{clean_prefix.title()}_{dominant_type.value.title()}"
        
        # Fallback naming
        return f"Cluster_{dominant_type.value.title()}"
    
    def _get_dominant_file_type(self, cluster_files: List[FileInfo]) -> FileType:
        """Get the most common file type in cluster"""
        file_types = [f.file_type for f in cluster_files]
        type_counts = Counter(file_types)
        return type_counts.most_common(1)[0][0]
    
    def _find_cluster_patterns(self, cluster_files: List[FileInfo]) -> List[str]:
        """Find common patterns in cluster files"""
        patterns = []
        
        # Check for common extensions
        extensions = [f.extension for f in cluster_files]
        ext_counts = Counter(extensions)
        if len(ext_counts) == 1:
            patterns.append(f"all_{list(ext_counts.keys())[0]}_files")
        
        # Check for size similarity
        sizes = [f.size for f in cluster_files]
        if len(sizes) > 1:
            size_cv = np.std(sizes) / np.mean(sizes)  # Coefficient of variation
            if size_cv < 0.5:  # Similar sizes
                patterns.append("similar_size")
        
        # Check for date similarity
        dates = [f.modified_at for f in cluster_files]
        if len(dates) > 1:
            date_range = max(dates) - min(dates)
            if date_range < 7 * 24 * 3600:  # Within a week
                patterns.append("created_recently")
        
        return patterns
    
    def _get_date_range(self, cluster_files: List[FileInfo]) -> Dict[str, float]:
        """Get date range for cluster files"""
        dates = [f.modified_at for f in cluster_files]
        return {
            'min_date': min(dates),
            'max_date': max(dates),
            'span_days': (max(dates) - min(dates)) / (24 * 3600)
        }
    
    def _generate_clustering_actions(self, files: List[FileInfo], labels: np.ndarray,
                                   cluster_analysis: Dict[int, Dict[str, Any]]) -> List[Action]:
        """Generate organization actions based on clustering results"""
        
        actions = []
        
        # Create directories for each cluster
        for cluster_id, analysis in cluster_analysis.items():
            folder_name = f"Clustered/{analysis['name']}"
            actions.append(Action(
                action_type=ActionType.CREATE_DIR,
                source="",
                destination=folder_name,
                reason=f"ML cluster: {analysis['size']} files, type: {analysis['dominant_type'].value}"
            ))
        
        # Create move actions for clustered files
        for i, file_info in enumerate(files):
            label = labels[i]
            
            if label == -1:  # Noise/outlier files
                target_folder = "Clustered/Uncategorized"
                reason = "ML clustering: outlier file"
                confidence = 0.3
            elif label in cluster_analysis:
                analysis = cluster_analysis[label]
                target_folder = f"Clustered/{analysis['name']}"
                reason = f"ML clustering: grouped with {analysis['size']} similar files"
                confidence = 0.8
            else:
                # Small cluster, move to uncategorized
                target_folder = "Clustered/Small_Groups"
                reason = "ML clustering: small group"
                confidence = 0.5
            
            # Ensure target folder action exists
            folder_action_exists = any(
                action.action_type == ActionType.CREATE_DIR and action.destination == target_folder
                for action in actions
            )
            if not folder_action_exists:
                actions.append(Action(
                    action_type=ActionType.CREATE_DIR,
                    source="",
                    destination=target_folder,
                    reason="ML clustering directory"
                ))
            
            # Add move action
            actions.append(Action(
                action_type=ActionType.MOVE,
                source=file_info.path,
                destination=f"{target_folder}/{file_info.name}",
                reason=reason,
                confidence=confidence
            ))
        
        return actions
    
    def _fallback_organization(self, files: List[FileInfo]) -> OrganizationPlan:
        """Fallback organization when clustering fails"""
        
        actions = []
        
        # Simple type-based organization
        type_folders = {}
        for file_info in files:
            folder_name = f"ByType/{file_info.file_type.value.title()}"
            
            if folder_name not in type_folders:
                type_folders[folder_name] = []
                actions.append(Action(
                    action_type=ActionType.CREATE_DIR,
                    source="",
                    destination=folder_name,
                    reason="Fallback: organize by file type"
                ))
            
            actions.append(Action(
                action_type=ActionType.MOVE,
                source=file_info.path,
                destination=f"{folder_name}/{file_info.name}",
                reason=f"Fallback: {file_info.file_type.value} file",
                confidence=0.6
            ))
        
        return OrganizationPlan(
            actions=actions,
            source_directory="",
            target_directory="organized"
        )
    
    def get_cluster_visualization_data(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Get data for visualizing clusters (useful for GUI)"""
        
        if len(files) < 2:
            return {'error': 'Too few files for clustering'}
        
        try:
            # Extract features
            feature_matrix, feature_names = self._extract_clustering_features(files)
            
            if feature_matrix.shape[1] == 0:
                return {'error': 'No features could be extracted'}
            
            # Perform clustering
            labels = self._perform_clustering(feature_matrix)
            
            # Dimensionality reduction for visualization
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize and reduce to 2D
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            pca = PCA(n_components=min(2, feature_matrix.shape[1]))
            reduced_features = pca.fit_transform(scaled_features)
            
            # Prepare visualization data
            viz_data = {
                'points': [
                    {
                        'x': float(reduced_features[i, 0]),
                        'y': float(reduced_features[i, 1]) if reduced_features.shape[1] > 1 else 0,
                        'file_name': files[i].name,
                        'file_type': files[i].file_type.value,
                        'cluster': int(labels[i]),
                        'size': files[i].size_mb
                    }
                    for i in range(len(files))
                ],
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'n_noise': list(labels).count(-1),
                'algorithm': self.config.algorithm,
                'explained_variance': pca.explained_variance_ratio_.tolist() if hasattr(pca, 'explained_variance_ratio_') else []
            }
            
            return viz_data
            
        except Exception as e:
            logger.error(f"Visualization data generation failed: {e}")
            return {'error': str(e)}