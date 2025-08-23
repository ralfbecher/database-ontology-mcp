"""Database Ontology MCP Server - MCP server for database ad hoc analysis with ontology support and interactive charting."""

__version__ = "0.3.0"
__author__ = "Database Ontology MCP Contributors"
__email__ = "contributors@example.com"
__description__ = "MCP server for database ad hoc analysis with ontology support and interactive charting"
__name__ = "Database Ontology MCP Server"

# Export main components for easier imports
from .database_manager import DatabaseManager, TableInfo, ColumnInfo
from .ontology_generator import OntologyGenerator
from .config import config_manager
from .constants import SUPPORTED_DB_TYPES

__all__ = [
    "DatabaseManager",
    "TableInfo", 
    "ColumnInfo",
    "OntologyGenerator",
    "config_manager",
    "SUPPORTED_DB_TYPES",
    "__version__",
    "__name__",
    "__description__",
]
