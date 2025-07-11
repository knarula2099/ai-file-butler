# file_butler/utils/config.py
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for File Butler"""
    
    DEFAULT_CONFIG = {
        'llm': {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'max_tokens': 500,
            'temperature': 0.2,
            'max_file_size_kb': 50,
            'batch_size': 5,
            'enable_caching': True,
            'cost_limit_usd': 10.0
        },
        'clustering': {
            'algorithm': 'dbscan',
            'min_samples': 2,
            'eps': 0.5,
            'max_features': 1000,
            'use_tfidf': True
        },
        'general': {
            'default_dry_run': True,
            'backup_mode': False,
            'max_file_size_mb': 100,
            'enable_duplicate_detection': True,
            'log_level': 'INFO'
        },
        'safety': {
            'protected_directories': [
                '/System', '/usr', '/bin', '/sbin',
                'C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)',
                '/Applications', '/Library/System'
            ],
            'skip_system_files': True,
            'confirm_deletions': True
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        possible_locations = [
            '.file_butler.yaml',
            '.file_butler.yml',
            '~/.file_butler.yaml',
            '~/.config/file_butler/config.yaml'
        ]
        
        for location in possible_locations:
            path = Path(location).expanduser()
            if path.exists():
                return str(path)
        
        return '.file_butler.yaml'  # Default location
    
    def load_config(self):
        """Load configuration from file"""
        try:
            config_path = Path(self.config_file).expanduser()
            if config_path.exists():
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                
                if user_config:
                    self._merge_config(self.config, user_config)
                    logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_file}: {e}")
        
        # Override with environment variables
        self._load_env_overrides()
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mappings = {
            'OPENAI_API_KEY': ('llm', 'api_key'),
            'ANTHROPIC_API_KEY': ('llm', 'anthropic_api_key'),
            'FILE_BUTLER_LLM_MODEL': ('llm', 'model'),
            'FILE_BUTLER_COST_LIMIT': ('llm', 'cost_limit_usd'),
            'FILE_BUTLER_LOG_LEVEL': ('general', 'log_level'),
            'FILE_BUTLER_DRY_RUN': ('general', 'default_dry_run'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if section not in self.config:
                    self.config[section] = {}
                
                # Type conversion
                if key in ['cost_limit_usd']:
                    value = float(value)
                elif key in ['default_dry_run']:
                    value = value.lower() in ('true', '1', 'yes')
                
                self.config[section][key] = value
    
    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        if key is None:
            return self.config.get(section, default)
        
        section_config = self.config.get(section, {})
        return section_config.get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_path = Path(self.config_file).expanduser()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def create_sample_config(self):
        """Create a sample configuration file"""
        sample_config = {
            'llm': {
                'provider': 'openai',  # or 'anthropic', 'local'
                'model': 'gpt-3.5-turbo',
                'cost_limit_usd': 5.0,
                'enable_caching': True
            },
            'general': {
                'default_dry_run': True,
                'log_level': 'INFO'
            },
            'safety': {
                'confirm_deletions': True,
                'skip_system_files': True
            }
        }
        
        try:
            config_path = Path(self.config_file).expanduser()
            with open(config_path, 'w') as f:
                f.write("# AI File Butler Configuration\n")
                f.write("# Remove this file to use defaults\n\n")
                yaml.dump(sample_config, f, default_flow_style=False, indent=2)
            
            print(f"Sample configuration created at {config_path}")
            print("Edit this file to customize File Butler behavior.")
        except Exception as e:
            logger.error(f"Failed to create sample config: {e}")

# Global config instance
_config = None

def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config