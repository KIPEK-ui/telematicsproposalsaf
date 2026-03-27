"""
Caching Utilities Module
Provides reusable caching infrastructure for the application.
Includes LRU cache with size limits, TTL-based caching, and persistence utilities.
"""

import hashlib
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple
from functools import wraps
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU Cache with size limits and optional TTL."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: Optional[int] = None):
        """
        Initialize LRU Cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live in seconds (None for no expiry)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order: list = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, respecting TTL."""
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        
        # Check TTL
        if self.ttl_seconds and (time.time() - timestamp) > self.ttl_seconds:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None
        
        # Update access order for LRU
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache, evicting oldest if needed."""
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        # Store with timestamp
        self._cache[key] = (value, time.time())
        
        # Evict oldest if over limit
        if len(self._cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted: {oldest_key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class CacheKey:
    """Utilities for generating deterministic cache keys."""
    
    @staticmethod
    def from_string(content: str) -> str:
        """Generate MD5 hash from string content."""
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def from_dict(data: Dict) -> str:
        """Generate MD5 hash from dictionary (JSON serialized)."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    @staticmethod
    def from_tuple(*args) -> str:
        """Generate MD5 hash from tuple of arguments."""
        content = "|".join(str(arg) for arg in args)
        return hashlib.md5(content.encode()).hexdigest()


class PersistentCache:
    """Disk-based persistent cache with automatic serialization."""
    
    def __init__(self, cache_dir: str = "data/.cache"):
        """
        Initialize persistent cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, cache_type: str = "pickle") -> Optional[Any]:
        """
        Retrieve value from disk cache.
        
        Args:
            key: Cache key
            cache_type: 'pickle' or 'json'
        
        Returns:
            Cached value or None if not found
        """
        try:
            ext = ".pkl" if cache_type == "pickle" else ".json"
            cache_file = self.cache_dir / f"{key}{ext}"
            
            if not cache_file.exists():
                return None
            
            if cache_type == "pickle":
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        except Exception as e:
            logger.warning(f"Failed to load cache {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, cache_type: str = "pickle") -> bool:
        """
        Store value to disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: 'pickle' or 'json'
        
        Returns:
            True if successful
        """
        try:
            ext = ".pkl" if cache_type == "pickle" else ".json"
            cache_file = self.cache_dir / f"{key}{ext}"
            
            if cache_type == "pickle":
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            else:
                with open(cache_file, 'w') as f:
                    json.dump(value, f)
            
            logger.debug(f"Cached to disk: {key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save cache {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cache file."""
        try:
            for ext in [".pkl", ".json"]:
                cache_file = self.cache_dir / f"{key}{ext}"
                if cache_file.exists():
                    cache_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache {key}: {e}")
            return False
    
    def clear_all(self) -> None:
        """Clear all cache files."""
        try:
            for cache_file in self.cache_dir.glob("*"):
                cache_file.unlink()
            logger.info("Cleared all disk cache")
        except Exception as e:
            logger.error(f"Failed to clear cache directory: {e}")


def cache_with_ttl(ttl_seconds: int = 300, max_size: int = 100):
    """
    Decorator for caching function results with TTL.
    
    Args:
        ttl_seconds: Time-to-live in seconds
        max_size: Maximum cache size
    
    Usage:
        @cache_with_ttl(ttl_seconds=300, max_size=50)
        def expensive_function(param1, param2):
            return result
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = CacheKey.from_tuple(func.__name__, args, kwargs)
            
            # Try cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {func.__name__}")
                return cached_result
            
            # Compute result
            logger.debug(f"Cache miss: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result)
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


def cache_persistent(cache_dir: str = "data/.cache", cache_type: str = "pickle"):
    """
    Decorator for persistent disk-based caching.
    
    Args:
        cache_dir: Directory for cache storage
        cache_type: 'pickle' or 'json'
    
    Usage:
        @cache_persistent(cache_type='pickle')
        def slow_operation(param1):
            return result
    """
    def decorator(func: Callable) -> Callable:
        persistent_cache = PersistentCache(cache_dir)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = CacheKey.from_tuple(func.__name__, args, kwargs)
            
            # Try persistent cache
            cached_result = persistent_cache.get(cache_key, cache_type=cache_type)
            if cached_result is not None:
                logger.debug(f"Persistent cache hit: {func.__name__}")
                return cached_result
            
            # Compute result
            logger.debug(f"Persistent cache miss: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            persistent_cache.set(cache_key, result, cache_type=cache_type)
            return result
        
        wrapper.cache = persistent_cache
        return wrapper
    
    return decorator
