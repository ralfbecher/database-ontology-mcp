"""Database Ontology MCP Server - Enhanced version with security, performance, and reliability improvements."""

__version__ = "0.2.0"
__author__ = "Database Ontology MCP Contributors"
__email__ = "contributors@example.com"
__description__ = "Enhanced MCP server for database schema analysis and ontology generation"

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
]
