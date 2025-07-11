# file_butler/utils/caching.py
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class LLMCache:
    """SQLite-based cache for LLM responses"""
    
    def __init__(self, cache_dir: str = ".file_butler_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "llm_cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    expires_at REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON llm_cache(expires_at)
            """)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, expires_at FROM llm_cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    value_str, expires_at = row
                    
                    # Check expiration
                    if expires_at and time.time() > expires_at:
                        self.delete(key)
                        return None
                    
                    return json.loads(value_str)
                
                return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set cached value"""
        try:
            expires_at = None
            if ttl_seconds:
                expires_at = time.time() + ttl_seconds
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache (key, value, timestamp, expires_at) VALUES (?, ?, ?, ?)",
                    (key, json.dumps(value), time.time(), expires_at)
                )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete cached value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
    
    def clear_expired(self):
        """Clear expired cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM llm_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (time.time(),)
                )
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM llm_cache")
                total_entries = cursor.fetchone()[0]
                
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM llm_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (time.time(),)
                )
                expired_entries = cursor.fetchone()[0]
                
                return {
                    'total_entries': total_entries,
                    'expired_entries': expired_entries,
                    'valid_entries': total_entries - expired_entries
                }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {'total_entries': 0, 'expired_entries': 0, 'valid_entries': 0}
