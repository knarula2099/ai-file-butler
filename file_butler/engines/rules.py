# file_butler/engines/rules.py
from typing import List, Dict, Any
from pathlib import Path
import re
from datetime import datetime, timedelta

from ..core.models import FileInfo, Action, ActionType, OrganizationPlan, FileType

class RuleBasedEngine:
    """Rule-based file organization engine"""
    
    def __init__(self, target_directory: str = "organized"):
        self.target_directory = target_directory
        self.rules = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default organization rules"""
        self.rules = [
            # System and junk files
            self._rule_system_files,
            self._rule_temp_files,
            self._rule_duplicate_patterns,
            
            # Documents by content type
            self._rule_financial_documents,
            self._rule_work_documents,
            self._rule_personal_documents,
            self._rule_educational_documents,
            
            # Images by type and pattern
            self._rule_screenshots,
            self._rule_camera_photos,
            self._rule_social_media_images,
            
            # Media files
            self._rule_videos,
            self._rule_audio,
            
            # Software and archives
            self._rule_installers,
            self._rule_archives,
            self._rule_code_files,
            
            # Fallback rules
            self._rule_by_file_type,
            self._rule_by_age,
        ]
    
    def organize(self, files: List[FileInfo]) -> OrganizationPlan:
        """Generate organization plan using rules"""
        actions = []
        
        # Create base directory structure
        base_dirs = self._get_base_directories()
        for dir_path in base_dirs:
            actions.append(Action(
                action_type=ActionType.CREATE_DIR,
                source="",
                destination=str(Path(self.target_directory) / dir_path)
            ))
        
        # Apply rules to each file
        for file_info in files:
            target_path = self._apply_rules(file_info)
            if target_path:
                full_target = str(Path(self.target_directory) / target_path)
                actions.append(Action(
                    action_type=ActionType.MOVE,
                    source=file_info.path,
                    destination=full_target,
                    reason=f"Rule-based classification: {target_path}"
                ))
        
        return OrganizationPlan(
            actions=actions,
            source_directory="",  # Will be set by caller
            target_directory=self.target_directory
        )
    
    def _get_base_directories(self) -> List[str]:
        """Get list of base directories to create"""
        return [
            "Documents/Work",
            "Documents/Personal", 
            "Documents/Financial",
            "Documents/Educational",
            "Images/Screenshots",
            "Images/Photos/Camera",
            "Images/Photos/Social_Media",
            "Images/Graphics",
            "Media/Videos",
            "Media/Audio",
            "Software/Installers",
            "Software/Archives",
            "Code",
            "System_Files",
            "Archive/Old_Files",
            "_Needs_Review"
        ]
    
    def _apply_rules(self, file_info: FileInfo) -> str:
        """Apply rules to determine target path for a file"""
        for rule in self.rules:
            result = rule(file_info)
            if result:
                return result
        
        # Default fallback
        return f"_Needs_Review/{file_info.name}"
    
    # Rule implementations
    def _rule_system_files(self, file_info: FileInfo) -> str:
        """Rule for system and hidden files"""
        name_lower = file_info.name.lower()
        
        system_files = [
            '.ds_store', 'thumbs.db', 'desktop.ini', '.localized',
            'icon\r', 'folder.jpg', 'albumartsmall.jpg'
        ]
        
        if name_lower in system_files or file_info.name.startswith('.'):
            return f"System_Files/{file_info.name}"
        
        return None
    
    def _rule_temp_files(self, file_info: FileInfo) -> str:
        """Rule for temporary files"""
        name_lower = file_info.name.lower()
        
        temp_patterns = [
            'untitled', 'new text document', 'new document',
            'copy of copy', 'temp', 'tmp', '~'
        ]
        
        # Check for temp file patterns
        for pattern in temp_patterns:
            if pattern in name_lower:
                return f"System_Files/Temp/{file_info.name}"
        
        # Check for backup file extensions
        if file_info.extension in ['.tmp', '.temp', '.bak', '.old']:
            return f"System_Files/Temp/{file_info.name}"
        
        return None
    
    def _rule_duplicate_patterns(self, file_info: FileInfo) -> str:
        """Rule for files with duplicate-like patterns"""
        name_lower = file_info.name.lower()
        
        # Common duplicate patterns
        duplicate_patterns = [
            r'\(\d+\)',  # filename (1), filename (2)
            r'_copy', r'_copy\d*',  # filename_copy, filename_copy2
            r'copy of',  # copy of filename
        ]
        
        name_without_ext = Path(file_info.name).stem.lower()
        
        for pattern in duplicate_patterns:
            if re.search(pattern, name_without_ext):
                return f"_Needs_Review/Potential_Duplicates/{file_info.name}"
        
        return None
    
    def _rule_financial_documents(self, file_info: FileInfo) -> str:
        """Rule for financial documents"""
        if file_info.file_type != FileType.DOCUMENT:
            return None
        
        name_lower = file_info.name.lower()
        
        financial_keywords = [
            'invoice', 'receipt', 'bill', 'statement', 'tax',
            'budget', 'expense', 'income', 'salary', 'pay',
            'bank', 'credit', 'loan', 'mortgage', 'insurance',
            'investment', 'portfolio', 'retirement', '401k'
        ]
        
        # Check filename and content keywords
        for keyword in financial_keywords:
            if keyword in name_lower:
                return f"Documents/Financial/{file_info.name}"
        
        # Check for content keywords if available
        if hasattr(file_info, 'features') and file_info.features:
            text_features = file_info.features.get('textcontent', {})
            keywords = text_features.get('top_keywords', [])
            
            for keyword in keywords:
                if keyword.lower() in financial_keywords:
                    return f"Documents/Financial/{file_info.name}"
        
        return None
    
    def _rule_work_documents(self, file_info: FileInfo) -> str:
        """Rule for work-related documents"""
        if file_info.file_type != FileType.DOCUMENT:
            return None
        
        name_lower = file_info.name.lower()
        
        work_keywords = [
            'meeting', 'agenda', 'project', 'report', 'proposal',
            'presentation', 'contract', 'agreement', 'memo',
            'quarterly', 'annual', 'performance', 'review',
            'team', 'employee', 'client', 'customer', 'business'
        ]
        
        for keyword in work_keywords:
            if keyword in name_lower:
                return f"Documents/Work/{file_info.name}"
        
        return None
    
    def _rule_personal_documents(self, file_info: FileInfo) -> str:
        """Rule for personal documents"""
        if file_info.file_type != FileType.DOCUMENT:
            return None
        
        name_lower = file_info.name.lower()
        
        personal_keywords = [
            'recipe', 'workout', 'health', 'medical', 'doctor',
            'vacation', 'travel', 'trip', 'hotel', 'flight',
            'family', 'wedding', 'birthday', 'anniversary',
            'hobby', 'book', 'reading', 'todo', 'list',
            'personal', 'diary', 'journal', 'note'
        ]
        
        for keyword in personal_keywords:
            if keyword in name_lower:
                return f"Documents/Personal/{file_info.name}"
        
        return None
    
    def _rule_educational_documents(self, file_info: FileInfo) -> str:
        """Rule for educational documents"""
        if file_info.file_type != FileType.DOCUMENT:
            return None
        
        name_lower = file_info.name.lower()
        
        education_keywords = [
            'course', 'lesson', 'tutorial', 'study', 'notes',
            'learning', 'education', 'school', 'university',
            'certificate', 'diploma', 'degree', 'training',
            'programming', 'python', 'javascript', 'machine learning',
            'data science', 'statistics', 'math', 'algorithm'
        ]
        
        for keyword in education_keywords:
            if keyword in name_lower:
                return f"Documents/Educational/{file_info.name}"
        
        return None
    
    def _rule_screenshots(self, file_info: FileInfo) -> str:
        """Rule for screenshots"""
        if file_info.file_type != FileType.IMAGE:
            return None
        
        name_lower = file_info.name.lower()
        
        screenshot_patterns = [
            'screenshot', 'screen shot', 'screen capture',
            'capture', 'snip', 'grab'
        ]
        
        for pattern in screenshot_patterns:
            if pattern in name_lower:
                return f"Images/Screenshots/{file_info.name}"
        
        # Check for screenshot filename patterns
        if re.search(r'screenshot.*\d{4}-\d{2}-\d{2}', name_lower):
            return f"Images/Screenshots/{file_info.name}"
        
        return None
    
    def _rule_camera_photos(self, file_info: FileInfo) -> str:
        """Rule for camera photos"""
        if file_info.file_type != FileType.IMAGE:
            return None
        
        name_lower = file_info.name.lower()
        
        # Common camera filename patterns
        camera_patterns = [
            r'^img_\d+',  # IMG_1234
            r'^dsc\d+',   # DSC1234  
            r'^dcim',     # DCIM files
            r'^\d{8}_\d+',  # 20240315_123456
        ]
        
        for pattern in camera_patterns:
            if re.search(pattern, name_lower):
                return f"Images/Photos/Camera/{file_info.name}"
        
        return None
    
    def _rule_social_media_images(self, file_info: FileInfo) -> str:
        """Rule for social media images"""
        if file_info.file_type != FileType.IMAGE:
            return None
        
        name_lower = file_info.name.lower()
        
        social_keywords = [
            'instagram', 'facebook', 'twitter', 'linkedin',
            'snapchat', 'tiktok', 'social', 'profile',
            'avatar', 'cover', 'story', 'post', 'meme'
        ]
        
        for keyword in social_keywords:
            if keyword in name_lower:
                return f"Images/Photos/Social_Media/{file_info.name}"
        
        return None
    
    def _rule_videos(self, file_info: FileInfo) -> str:
        """Rule for video files"""
        if file_info.file_type != FileType.VIDEO:
            return None
        
        name_lower = file_info.name.lower()
        
        # Categorize by content type
        if any(word in name_lower for word in ['tutorial', 'course', 'lesson', 'education']):
            return f"Media/Videos/Educational/{file_info.name}"
        elif any(word in name_lower for word in ['meeting', 'conference', 'presentation']):
            return f"Media/Videos/Work/{file_info.name}"
        elif any(word in name_lower for word in ['movie', 'film', 'show', 'entertainment']):
            return f"Media/Videos/Entertainment/{file_info.name}"
        else:
            return f"Media/Videos/{file_info.name}"
    
    def _rule_audio(self, file_info: FileInfo) -> str:
        """Rule for audio files"""
        if file_info.file_type != FileType.AUDIO:
            return None
        
        name_lower = file_info.name.lower()
        
        # Categorize by content type
        if any(word in name_lower for word in ['podcast', 'interview', 'talk']):
            return f"Media/Audio/Podcasts/{file_info.name}"
        elif any(word in name_lower for word in ['song', 'music', 'album', 'track']):
            return f"Media/Audio/Music/{file_info.name}"
        elif any(word in name_lower for word in ['meeting', 'recording', 'note']):
            return f"Media/Audio/Recordings/{file_info.name}"
        else:
            return f"Media/Audio/{file_info.name}"
    
    def _rule_installers(self, file_info: FileInfo) -> str:
        """Rule for software installers"""
        installer_extensions = ['.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm']
        
        if file_info.extension.lower() in installer_extensions:
            return f"Software/Installers/{file_info.name}"
        
        name_lower = file_info.name.lower()
        if any(word in name_lower for word in ['installer', 'setup', 'install']):
            return f"Software/Installers/{file_info.name}"
        
        return None
    
    def _rule_archives(self, file_info: FileInfo) -> str:
        """Rule for archive files"""
        if file_info.file_type != FileType.ARCHIVE:
            return None
        
        name_lower = file_info.name.lower()
        
        # Categorize by content
        if any(word in name_lower for word in ['backup', 'archive', 'old']):
            return f"Software/Archives/Backups/{file_info.name}"
        elif any(word in name_lower for word in ['project', 'source', 'code']):
            return f"Software/Archives/Projects/{file_info.name}"
        else:
            return f"Software/Archives/{file_info.name}"
    
    def _rule_code_files(self, file_info: FileInfo) -> str:
        """Rule for code files"""
        if file_info.file_type != FileType.CODE:
            return None
        
        name_lower = file_info.name.lower()
        
        # Categorize by language/type
        if file_info.extension in ['.py']:
            return f"Code/Python/{file_info.name}"
        elif file_info.extension in ['.js', '.jsx', '.ts', '.tsx']:
            return f"Code/JavaScript/{file_info.name}"
        elif file_info.extension in ['.html', '.css']:
            return f"Code/Web/{file_info.name}"
        elif file_info.extension in ['.java']:
            return f"Code/Java/{file_info.name}"
        elif file_info.extension in ['.cpp', '.c', '.h']:
            return f"Code/C_CPP/{file_info.name}"
        else:
            return f"Code/Other/{file_info.name}"
    
    def _rule_by_file_type(self, file_info: FileInfo) -> str:
        """Fallback rule by file type"""
        
        # General categorization by file type
        if file_info.file_type == FileType.DOCUMENT:
            return f"Documents/General/{file_info.name}"
        elif file_info.file_type == FileType.IMAGE:
            return f"Images/General/{file_info.name}"
        elif file_info.file_type == FileType.VIDEO:
            return f"Media/Videos/{file_info.name}"
        elif file_info.file_type == FileType.AUDIO:
            return f"Media/Audio/{file_info.name}"
        elif file_info.file_type == FileType.ARCHIVE:
            return f"Software/Archives/{file_info.name}"
        elif file_info.file_type == FileType.CODE:
            return f"Code/{file_info.name}"
        else:
            return f"Other/{file_info.name}"
    
    def _rule_by_age(self, file_info: FileInfo) -> str:
        """Rule for organizing by age (for uncategorized files)"""
        
        # Only apply to files that haven't been categorized yet
        # This would be called after other rules fail
        
        age_days = file_info.age_days
        
        if age_days > 365:  # More than 1 year old
            year = datetime.fromtimestamp(file_info.modified_at).year
            return f"Archive/Old_Files/{year}/{file_info.name}"
        elif age_days > 90:  # More than 3 months old
            return f"Archive/Older/{file_info.name}"
        else:
            return None  # Let other rules handle recent files

    def add_custom_rule(self, rule_func, priority=None):
        """Add a custom rule function"""
        if priority is None:
            self.rules.append(rule_func)
        else:
            self.rules.insert(priority, rule_func)
    
    def remove_rule(self, rule_func):
        """Remove a rule function"""
        if rule_func in self.rules:
            self.rules.remove(rule_func)