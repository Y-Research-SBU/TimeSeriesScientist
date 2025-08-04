"""
Memory Management for Time Series Prediction Agent
时序预测代理系统的内存管理模块
"""

import json
import pickle
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


class ExperimentMemory:
    """
    Experiment Memory Management Class, used to share information and state between agents
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory = {}
        self.history = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "config": config,
            "version": "1.0.0"
        }
        
        # Initialize memory structure
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize memory structure"""
        self.memory = {
            "data": {},           # Raw data and preprocessed data
            "analysis": {},       # Data analysis results
            "models": {},         # Model related information
            "forecasts": {},      # Prediction results
            "metrics": {},        # Evaluation metrics
            "visualizations": {}, # Paths to visualization files
            "reports": {},        # Report content
            "errors": [],         # Error logs
            "warnings": [],       # Warning logs
            "timestamps": {}      # Timestamps for each stage
        }
    
    def store(self, key: str, value: Any, category: str = "general"):
        """Store data in memory"""
        if category not in self.memory:
            self.memory[category] = {}
        
        self.memory[category][key] = value
        self._add_timestamp(f"{category}.{key}")
    
    def retrieve(self, key: str, category: str = "general", default: Any = None):
        """Retrieve data from memory"""
        if category not in self.memory:
            return default
        return self.memory[category].get(key, default)
    
    def exists(self, key: str, category: str = "general") -> bool:
        """Check if data exists"""
        if category not in self.memory:
            return False
        return key in self.memory[category]
    
    def delete(self, key: str, category: str = "general"):
        """Delete data from memory"""
        if category in self.memory and key in self.memory[category]:
            del self.memory[category][key]
    
    def clear_category(self, category: str):
        """Clear all data for a specific category"""
        if category in self.memory:
            self.memory[category] = {}
    
    def clear_all(self):
        """Clear all memory data"""
        self._initialize_memory()
    
    def get_category(self, category: str) -> Dict[str, Any]:
        """Get all data for a specific category"""
        return self.memory.get(category, {})
    
    def add_history(self, action: str, data: Any = None, metadata: Dict[str, Any] = None):
        """Add history record"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data,
            "metadata": metadata or {}
        }
        self.history.append(history_entry)
    
    def get_history(self, action: str = None) -> List[Dict[str, Any]]:
        """Get history records"""
        if action is None:
            return self.history
        return [entry for entry in self.history if entry["action"] == action]
    
    def add_error(self, error: str, context: Dict[str, Any] = None):
        """Add error log"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context or {}
        }
        self.memory["errors"].append(error_entry)
    
    def add_warning(self, warning: str, context: Dict[str, Any] = None):
        """Add warning log"""
        warning_entry = {
            "timestamp": datetime.now().isoformat(),
            "warning": warning,
            "context": context or {}
        }
        self.memory["warnings"].append(warning_entry)
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all errors"""
        return self.memory["errors"]
    
    def get_warnings(self) -> List[Dict[str, Any]]:
        """Get all warnings"""
        return self.memory["warnings"]
    
    def _add_timestamp(self, key: str):
        """Add timestamp"""
        self.memory["timestamps"][key] = datetime.now().isoformat()
    
    def get_timestamp(self, key: str) -> Optional[str]:
        """Get timestamp"""
        return self.memory["timestamps"].get(key)
    
    def save_to_file(self, filepath: str, format: str = "json"):
        """Save memory to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            # Handle non-serializable objects
            serializable_memory = self._make_serializable(self.memory)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_memory, f, indent=2, ensure_ascii=False)
        elif format.lower() == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(self.memory, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_from_file(self, filepath: str, format: str = "json"):
        """Load memory from file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Memory file not found: {filepath}")
        
        if format.lower() == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                self.memory = json.load(f)
        elif format.lower() == "pickle":
            with open(filepath, 'rb') as f:
                self.memory = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _make_serializable(self, obj):
        """Convert object to a serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory summary"""
        summary = {
            "total_entries": sum(len(category) for category in self.memory.values() if isinstance(category, dict)),
            "categories": {k: len(v) if isinstance(v, dict) else len(v) for k, v in self.memory.items()},
            "history_count": len(self.history),
            "error_count": len(self.memory["errors"]),
            "warning_count": len(self.memory["warnings"]),
            "created_at": self.metadata["created_at"],
            "last_updated": max(self.memory["timestamps"].values()) if self.memory["timestamps"] else None
        }
        return summary
    
    def __str__(self) -> str:
        """String representation"""
        summary = self.get_summary()
        return f"ExperimentMemory(total_entries={summary['total_entries']}, categories={summary['categories']})"
    
    def __repr__(self) -> str:
        return self.__str__()


class StateManager:
    """
    State Manager, used to manage experiment state
    """
    
    def __init__(self, memory: ExperimentMemory):
        self.memory = memory
        self.current_state = {}
    
    def set_state(self, state: Dict[str, Any]):
        """Set current state"""
        self.current_state = state.copy()
        self.memory.store("current_state", state, "system")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.current_state.copy()
    
    def update_state(self, updates: Dict[str, Any]):
        """Update state"""
        self.current_state.update(updates)
        self.memory.store("current_state", self.current_state, "system")
    
    def get_state_value(self, key: str, default: Any = None):
        """Get a specific value from the state"""
        return self.current_state.get(key, default)
    
    def set_state_value(self, key: str, value: Any):
        """Set a specific value in the state"""
        self.current_state[key] = value
        self.memory.store("current_state", self.current_state, "system")


class CacheManager:
    """
    Cache Manager, used to manage temporary data and calculation results
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.cache_metadata = {}
        self.max_cache_size = config.get("max_cache_size", 1000)
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set cache"""
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.cache_metadata[key] = {
            "created_at": datetime.now().isoformat(),
            "ttl": ttl,
            "size": self._estimate_size(value)
        }
    
    def get(self, key: str, default: Any = None):
        """Get cache"""
        if key not in self.cache:
            return default
        
        # Check TTL
        metadata = self.cache_metadata.get(key, {})
        if metadata.get("ttl"):
            created_at = datetime.fromisoformat(metadata["created_at"])
            if (datetime.now() - created_at).seconds > metadata["ttl"]:
                self.delete(key)
                return default
        
        return self.cache[key]
    
    def delete(self, key: str):
        """Delete cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.cache_metadata:
            del self.cache_metadata[key]
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.cache_metadata.clear()
    
    def exists(self, key: str) -> bool:
        """Check if cache exists"""
        return key in self.cache
    
    def _evict_oldest(self):
        """Evict the oldest cache item"""
        if not self.cache:
            return
        
        oldest_key = min(self.cache_metadata.keys(), 
                        key=lambda k: self.cache_metadata[k]["created_at"])
        self.delete(oldest_key)
    
    def _estimate_size(self, obj) -> int:
        """Estimate object size"""
        try:
            return len(str(obj))
        except:
            return 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_items": len(self.cache),
            "total_size": sum(meta.get("size", 0) for meta in self.cache_metadata.values()),
            "max_size": self.max_cache_size
        } 