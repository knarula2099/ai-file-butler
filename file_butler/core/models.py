# file_butler/core/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path
import time

class ActionType(Enum):
    """Types of file operations"""
    MOVE = "move"
    COPY = "copy"
    DELETE = "delete"
    CREATE_DIR = "create_dir"
    RENAME = "rename"

class FileType(Enum):
    """File type categories"""
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    CODE = "code"
    OTHER = "other"

@dataclass
class FileInfo:
    """Represents a file and its metadata"""
    path: str
    name: str
    extension: str
    size: int
    created_at: float
    modified_at: float
    accessed_at: float
    
    # Extended attributes
    parent_dir: str = field(init=False)
    file_type: FileType = field(init=False)
    features: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.parent_dir = str(Path(self.path).parent)
        self.file_type = self._determine_file_type()
    
    def _determine_file_type(self) -> FileType:
        """Determine file type based on extension"""
        ext = self.extension.lower()
        
        if ext in {'.txt', '.md', '.pdf', '.doc', '.docx', '.rtf', '.odt'}:
            return FileType.DOCUMENT
        elif ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}:
            return FileType.IMAGE
        elif ext in {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}:
            return FileType.VIDEO
        elif ext in {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'}:
            return FileType.AUDIO
        elif ext in {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'}:
            return FileType.ARCHIVE
        elif ext in {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php'}:
            return FileType.CODE
        else:
            return FileType.OTHER
    
    @property
    def size_mb(self) -> float:
        """File size in megabytes"""
        return self.size / (1024 * 1024)
    
    @property
    def age_days(self) -> float:
        """File age in days since last modification"""
        return (time.time() - self.modified_at) / (24 * 3600)

@dataclass
class Action:
    """Represents a planned file operation"""
    action_type: ActionType
    source: str
    destination: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.action_type == ActionType.MOVE:
            return f"MOVE: {Path(self.source).name} → {self.destination}"
        elif self.action_type == ActionType.DELETE:
            return f"DELETE: {Path(self.source).name} ({self.reason})"
        elif self.action_type == ActionType.CREATE_DIR:
            return f"CREATE: {self.destination}/"
        else:
            return f"{self.action_type.value.upper()}: {self.source}"

@dataclass
class OrganizationPlan:
    """Complete organization plan with actions and statistics"""
    actions: List[Action]
    source_directory: str
    target_directory: str
    created_at: float = field(default_factory=time.time)
    
    @property
    def move_actions(self) -> List[Action]:
        return [a for a in self.actions if a.action_type == ActionType.MOVE]
    
    @property
    def delete_actions(self) -> List[Action]:
        return [a for a in self.actions if a.action_type == ActionType.DELETE]
    
    @property
    def create_dir_actions(self) -> List[Action]:
        return [a for a in self.actions if a.action_type == ActionType.CREATE_DIR]
    
    def summary(self) -> Dict[str, int]:
        """Get summary statistics of the plan"""
        return {
            "total_actions": len(self.actions),
            "files_to_move": len(self.move_actions),
            "files_to_delete": len(self.delete_actions),
            "directories_to_create": len(self.create_dir_actions)
        }

@dataclass
class ScanResult:
    """Result of directory scanning"""
    files: List[FileInfo]
    source_directory: str
    scan_time: float
    errors: List[str] = field(default_factory=list)
    
    @property
    def total_size_mb(self) -> float:
        return sum(f.size_mb for f in self.files)
    
    @property
    def file_types_distribution(self) -> Dict[FileType, int]:
        distribution = {}
        for file_info in self.files:
            distribution[file_info.file_type] = distribution.get(file_info.file_type, 0) + 1
        return distribution