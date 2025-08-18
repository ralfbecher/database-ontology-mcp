"""Simple file-based connection parameter cache for MCP tool persistence."""

import json
import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Store connection params in project directory (accessible within working dir)
CACHE_FILE = Path(__file__).parent.parent / ".mcp_db_connection.json"

def save_connection_params(params: Dict[str, Any]) -> None:
    """Save database connection parameters to cache file."""
    try:
        # Remove password from logs but keep in cache
        safe_params = {k: v for k, v in params.items()}
        logger.info(f"Saving connection parameters for {params.get('type', 'unknown')} database")
        
        with open(CACHE_FILE, 'w') as f:
            json.dump(params, f)
        
        logger.debug(f"Connection parameters saved to {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save connection parameters: {e}")

def load_connection_params() -> Optional[Dict[str, Any]]:
    """Load database connection parameters from cache file."""
    try:
        if not CACHE_FILE.exists():
            logger.debug("No cached connection parameters found")
            return None
        
        with open(CACHE_FILE, 'r') as f:
            params = json.load(f)
        
        logger.info(f"Loaded cached connection parameters for {params.get('type', 'unknown')} database")
        return params
    except Exception as e:
        logger.error(f"Failed to load connection parameters: {e}")
        return None

def clear_connection_cache() -> None:
    """Clear the connection parameter cache."""
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            logger.info("Connection cache cleared")
    except Exception as e:
        logger.error(f"Failed to clear connection cache: {e}")