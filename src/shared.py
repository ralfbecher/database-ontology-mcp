"""Shared instances and utilities for the Database Ontology MCP Server."""

import logging
from typing import Optional, Dict, Any

from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Global database manager instance shared across all tools
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def create_error_response(message: str, error_type: str, details: str = None) -> Dict[str, Any]:
    """Create a standardized error response."""
    error_response = {
        "success": False,
        "error": message,
        "error_type": error_type
    }
    if details:
        error_response["details"] = details
    return error_response