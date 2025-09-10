"""Simple file-based connection parameter cache for MCP tool persistence."""

import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Store connection params in project directory (accessible within working dir)
CACHE_FILE = Path(__file__).parent.parent / ".mcp_db_connection.json"


def save_connection_params(params: Dict[str, Any]) -> None:
    """Save database connection parameters (DEPRECATED - SECURITY RISK).

    This function is deprecated due to security concerns with storing
    credentials in plain text files. Use secure credential management instead.
    """
    logger.warning(
        "save_connection_params is deprecated due to security risks - "
        "credentials not saved"
    )

    # Only log non-sensitive connection info
    safe_params = {
        k: v for k, v in params.items()
        if k.lower() not in {
            'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'pat'
        }
    }
    logger.info(
        "Connection attempt for %s database: %s",
        params.get('type', 'unknown'),
        safe_params
    )



def load_connection_params() -> Optional[Dict[str, Any]]:
    """Load database connection parameters from cache file."""
    try:
        if not CACHE_FILE.exists():
            logger.debug("No cached connection parameters found")
            return None

        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            params: Dict[str, Any] = json.load(f)

        logger.info(
            "Loaded cached connection parameters for %s database",
            params.get('type', 'unknown')
        )
        return params
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load connection parameters: %s", e)
        return None


def clear_connection_cache() -> None:
    """Clear the connection parameter cache."""
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            logger.info("Connection cache cleared")
    except OSError as e:
        logger.error("Failed to clear connection cache: %s", e)
