"""Centralized cache module for tdprepview-mcp server.

This module provides a single Cache class that manages all cached data
including pipelines, models, and column information.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class Cache:
    """Centralized cache for all server data."""
    
    def __init__(self):
        # Pipeline cache: pipeline_id -> (list_of_feature_output_column_names, pipeline_dict)
        self.pipelines: Dict[str, Tuple[List[str], Dict[str, Any]]] = {}
        # prep view cache: full_view_name
        #           -> (pipeline_id, input_schema, input_table, output_feat_colnames, output_nonfeat_colnames)
        self.prep_views : Dict[str, Tuple[str,str,str,List[str],List[str]]] = {}
        
        # Model cache: model_id -> {'onnx_model': ..., 'onnx_path': ...,  'metadata': ...}
        self.models: Dict[str, Dict[str, Any]] = {}

        # deployments: model_id -> viewname
        self.deployments:  Dict[str, str] = {}





# Global cache instance - will be initialized when server starts
cache: Optional[Cache] = None


def get_cache() -> Cache:
    """Get the global cache instance."""
    global cache
    if cache is None:
        cache = Cache()
    return cache


def init_cache() -> Cache:
    """Initialize the global cache."""
    global cache
    cache = Cache()
    return cache