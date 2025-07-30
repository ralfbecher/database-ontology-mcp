"""Main MCP server application using FastMCP."""

import asyncio
import json
import logging
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from fastmcp import FastMCP
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .database_manager import DatabaseManager, TableInfo
from .ontology_generator import OntologyGenerator

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- MCP Server Setup ---

mcp = FastMCP("Database Ontology MCP Server")

# --- Dependency Management ---

class ServerState:
    """Manages server state and dependencies."""
    
    def __init__(self):
        self._db_manager: Optional[DatabaseManager] = None
        self._ontology_generator: Optional[OntologyGenerator] = None
    
    def get_db_manager(self) -> DatabaseManager:
        """Get or create database manager instance."""
        if self._db_manager is None:
            self._db_manager = DatabaseManager()
        return self._db_manager
    
    def get_ontology_generator(self, base_uri: str = "http://example.com/ontology/") -> OntologyGenerator:
        """Get ontology generator instance."""
        # Always create new instance to avoid state pollution
        return OntologyGenerator(base_uri=base_uri)
    
    def cleanup(self):
        """Clean up resources."""
        if self._db_manager:
            self._db_manager.disconnect()
            self._db_manager = None
        self._ontology_generator = None

# Global server state
_server_state = ServerState()

# --- Error Response Helper ---

class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: str
    error_type: str = "unknown"
    details: Optional[str] = None

def create_error_response(error_msg: str, error_type: str = "unknown", details: Optional[str] = None) -> str:
    """Create a standardized error response."""
    response = ErrorResponse(error=error_msg, error_type=error_type, details=details)
    return response.model_dump_json()

def safe_execute(func, *args, **kwargs):
    """Helper function to safely execute MCP tool functions with error handling."""
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        logger.error(f"Runtime error in {func.__name__}: {e}")
        return create_error_response(str(e), "runtime_error")
    except Exception as e:
        logger.error(f"Unexpected error in {func.__name__}: {e}")
        return create_error_response(f"Internal server error: {str(e)}", "internal_error")

# --- MCP Tools ---

@mcp.tool()
def connect_database(
    db_type: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    account: Optional[str] = None,
    warehouse: Optional[str] = None,
    schema: Optional[str] = "PUBLIC"
) -> str:
    """Connect to a database (PostgreSQL or Snowflake) and return connection status.
    
    Args:
        db_type: Database type - either 'postgresql' or 'snowflake'
        host: Database host (for PostgreSQL)
        port: Database port (for PostgreSQL)
        database: Database name
        username: Database username
        password: Database password
        account: Snowflake account identifier (for Snowflake)
        warehouse: Snowflake warehouse (for Snowflake)
        schema: Database schema (default: PUBLIC for Snowflake, public for PostgreSQL)
    
    Returns:
        Connection status message or error JSON
    """
    # Validate input parameters
    if not db_type or db_type not in ["postgresql", "snowflake"]:
        return create_error_response(
            f"Invalid database type '{db_type}'. Use 'postgresql' or 'snowflake'.",
            "validation_error"
        )
    
    db_manager = _server_state.get_db_manager()
    
    if db_type == "postgresql":
        # Validate required parameters for PostgreSQL
        required_params = {"host": host, "port": port, "database": database, "username": username, "password": password}
        missing_params = [k for k, v in required_params.items() if v is None]
        if missing_params:
            return create_error_response(
                f"Missing required parameters for PostgreSQL: {', '.join(missing_params)}",
                "validation_error"
            )
        
        success = db_manager.connect_postgresql(
            host=str(host),
            port=int(port),
            database=str(database),
            username=str(username),
            password=str(password)
        )
    elif db_type == "snowflake":
        # Validate required parameters for Snowflake
        required_params = {"account": account, "username": username, "password": password, "warehouse": warehouse, "database": database}
        missing_params = [k for k, v in required_params.items() if v is None]
        if missing_params:
            return create_error_response(
                f"Missing required parameters for Snowflake: {', '.join(missing_params)}",
                "validation_error"
            )
        
        success = db_manager.connect_snowflake(
            account=str(account),
            username=str(username),
            password=str(password),
            warehouse=str(warehouse),
            database=str(database),
            schema=schema or "PUBLIC"
        )

    if success:
        return f"Successfully connected to {db_type} database: {database}"
    else:
        return create_error_response(
            f"Failed to connect to {db_type} database: {database}",
            "connection_error"
        )


@mcp.tool()
def list_schemas() -> List[str]:
    """Get a list of available schemas from the connected database.
    
    Returns:
        List of schema names or error response
    """
    db_manager = _server_state.get_db_manager()
    schemas = db_manager.get_schemas()
    return schemas if schemas else []


@mcp.tool()
def analyze_schema(schema_name: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a database schema and return detailed table information.
    
    Args:
        schema_name: Name of the schema to analyze (optional)
    
    Returns:
        Dictionary containing tables and their detailed information or error response
    """
    db_manager = _server_state.get_db_manager()
    tables = db_manager.get_tables(schema_name)
    
    all_table_info = []
    for table_name in tables:
        table_info = db_manager.analyze_table(table_name, schema_name)
        if table_info:
            # Convert dataclass to dict for JSON serialization
            table_dict = {
                "name": table_info.name,
                "schema": table_info.schema,
                "columns": [
                    {
                        "name": col.name,
                        "data_type": col.data_type,
                        "is_nullable": col.is_nullable,
                        "is_primary_key": col.is_primary_key,
                        "is_foreign_key": col.is_foreign_key,
                        "foreign_key_table": col.foreign_key_table,
                        "foreign_key_column": col.foreign_key_column,
                        "comment": col.comment
                    } for col in table_info.columns
                ],
                "primary_keys": table_info.primary_keys,
                "foreign_keys": table_info.foreign_keys,
                "comment": table_info.comment,
                "row_count": table_info.row_count
            }
            all_table_info.append(table_dict)
    
    return {
        "schema": schema_name or "default",
        "table_count": len(all_table_info),
        "tables": all_table_info
    }


@mcp.tool()
def generate_ontology(
    schema_name: Optional[str] = None,
    base_uri: str = "http://example.com/ontology/",
    enrich_llm: bool = False
) -> str:
    """Generate an RDF ontology from the database schema.
    
    Args:
        schema_name: Name of the schema to generate ontology from (optional)
        base_uri: Base URI for the ontology (default: http://example.com/ontology/)
        enrich_llm: Whether to enrich the ontology with LLM insights (default: False)
    
    Returns:
        RDF ontology in Turtle format or error response
    """
    # Validate base_uri
    if not base_uri.endswith('/'):
        base_uri += '/'
    
    db_manager = _server_state.get_db_manager()
    tables = db_manager.get_tables(schema_name)

    tables_info = []
    for table_name in tables:
        table_info = db_manager.analyze_table(table_name, schema_name)
        if table_info:
            tables_info.append(table_info)

    if not tables_info:
        return create_error_response(
            f"No tables found in schema '{schema_name or 'default'}' to generate ontology",
            "data_error"
        )

    generator = _server_state.get_ontology_generator(base_uri=base_uri)
    ontology_ttl = generator.generate_from_schema(tables_info)

    if enrich_llm:
        data_samples = {}
        for table in tables_info:
            try:
                data_samples[table.name] = db_manager.sample_table_data(table.name, schema_name, limit=5)
            except Exception as e:
                logger.warning(f"Could not sample data from table {table.name}: {e}")
        
        ontology_ttl = generator.enrich_with_llm(tables_info, data_samples)

    return ontology_ttl


@mcp.tool()
def sample_table_data(
    table_name: str,
    schema_name: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Sample data from a specific table for analysis.
    
    Args:
        table_name: Name of the table to sample
        schema_name: Schema containing the table (optional)
        limit: Maximum number of rows to return (default: 10, max: 100)
    
    Returns:
        List of sample rows as dictionaries or error response
    """
    # Validate parameters
    if not table_name:
        return [{"error": "Table name is required"}]
    
    if limit <= 0 or limit > 100:
        limit = 10
    
    db_manager = _server_state.get_db_manager()
    sample_data = db_manager.sample_table_data(table_name, schema_name, limit)
    return sample_data


@mcp.tool()
def get_table_relationships(schema_name: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
    """Get foreign key relationships between tables in a schema.
    
    Args:
        schema_name: Name of the schema to analyze relationships (optional)
    
    Returns:
        Dictionary mapping table names to their foreign key relationships or error response
    """
    db_manager = _server_state.get_db_manager()
    relationships = db_manager.get_table_relationships(schema_name)
    return relationships


@mcp.tool()
def get_server_info() -> Dict[str, Any]:
    """Get information about the MCP server and its capabilities.
    
    Returns:
        Dictionary containing server information
    """
    return {
        "name": "Database Ontology MCP Server",
        "version": "0.1.0",
        "description": "MCP server for database schema analysis and ontology generation",
        "supported_databases": ["postgresql", "snowflake"],
        "features": [
            "Database connection management",
            "Schema analysis",
            "Table relationship mapping",
            "RDF/OWL ontology generation",
            "LLM-enhanced ontology enrichment"
        ],
        "tools": [
            "connect_database",
            "list_schemas", 
            "analyze_schema",
            "generate_ontology",
            "sample_table_data",
            "get_table_relationships",
            "get_server_info"
        ]
    }


# --- Cleanup on shutdown ---

def cleanup_server():
    """Clean up server resources."""
    _server_state.cleanup()


if __name__ == "__main__":
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    finally:
        cleanup_server()
