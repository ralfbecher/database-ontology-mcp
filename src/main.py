#!/usr/bin/env python3
"""
Streamlined Database Ontology MCP Server

A focused MCP server with 8 essential tools for database analysis with automatic ontology generation.
Main tool: get_analysis_context() - provides complete schema analysis with integrated ontology.
"""

import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Union

from fastmcp import FastMCP
from .database_manager import DatabaseManager, TableInfo, ColumnInfo
from .ontology_generator import OntologyGenerator
from .config import Config

# Initialize configuration and logging
config = Config()
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("Database Ontology MCP Server")

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def create_error_response(message: str, error_type: str, details: str = None) -> str:
    """Create a standardized error response."""
    error_response = {
        "error": message,
        "error_type": error_type
    }
    if details:
        error_response["details"] = details
    return json.dumps(error_response)

@contextmanager
def error_handler(operation_name: str):
    """Context manager for consistent error handling."""
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation_name}: {type(e).__name__}: {e}")
        raise

# =============================================================================
# 🚀 STREAMLINED MCP TOOLS (8 Essential Tools)
# =============================================================================

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
) -> Union[Dict[str, Any], str]:
    """Connect to a PostgreSQL or Snowflake database.
    
    Args:
        db_type: Database type ("postgresql" or "snowflake")
        host: Database host (PostgreSQL only)
        port: Database port (PostgreSQL only) 
        database: Database name
        username: Username for authentication
        password: Password for authentication
        account: Snowflake account identifier (Snowflake only)
        warehouse: Snowflake warehouse (Snowflake only)
        schema: Schema name (Snowflake only, default: "PUBLIC")
    
    Returns:
        Connection status information or error response
    """
    with error_handler("connect_database") as handler:
        db_manager = get_db_manager()
        
        try:
            if db_type.lower() == "postgresql":
                if not all([host, port, database, username]):
                    return create_error_response(
                        "Missing required PostgreSQL parameters: host, port, database, username",
                        "parameter_error"
                    )
                success = db_manager.connect_postgresql(host, port, database, username, password or "")
                
            elif db_type.lower() == "snowflake":
                if not all([account, username, database, warehouse]):
                    return create_error_response(
                        "Missing required Snowflake parameters: account, username, database, warehouse", 
                        "parameter_error"
                    )
                success = db_manager.connect_snowflake(account, username, password or "", warehouse, database, schema)
                
            else:
                return create_error_response(
                    f"Unsupported database type: {db_type}. Use 'postgresql' or 'snowflake'",
                    "parameter_error"
                )
            
            if success:
                return {
                    "success": True,
                    "message": f"Successfully connected to {db_type} database",
                    "connection_info": db_manager.connection_info
                }
            else:
                return create_error_response(
                    f"Failed to connect to {db_type} database",
                    "connection_error"
                )
                
        except Exception as e:
            logger.error(f"Connection attempt failed: {type(e).__name__}: {e}")
            return create_error_response(
                f"Connection failed: {type(e).__name__}",
                "connection_error",
                str(e)
            )

@mcp.tool()
def list_schemas() -> Union[List[str], str]:
    """Get a list of available schemas from the connected database.
    
    Returns:
        List of schema names or error response
    """
    with error_handler("list_schemas") as handler:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error", 
                "Use connect_database tool first"
            )
        
        try:
            schemas = db_manager.get_schemas()
            logger.debug(f"Retrieved {len(schemas)} schemas")
            return schemas
        except Exception as e:
            logger.error(f"Failed to list schemas: {e}")
            return create_error_response(
                "Failed to retrieve schema list",
                "database_error",
                str(e)
            )

@mcp.tool()
def get_analysis_context(
    schema_name: Optional[str] = None,
    include_ontology: bool = True
) -> Union[Dict[str, Any], str]:
    """🌟 MAIN TOOL: Get comprehensive analysis context for data exploration and SQL generation.
    
    This is the primary tool for database analysis. It provides everything needed in one call:
    - Complete schema structure (tables, columns, relationships)  
    - Automatic ontology generation with SQL references
    - Ready-to-use JOIN conditions and column references
    - Relationship warnings for safe aggregations
    - SQL generation hints and best practices
    
    Args:
        schema_name: Name of the schema to analyze (optional)
        include_ontology: Whether to generate ontology (default: True, recommended)
    
    Returns:
        Dictionary containing complete analysis context with schema and ontology data
    """
    try:
        db_manager = get_db_manager()
        
        # Check connection
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        logger.info(f"Generating complete analysis context for schema: {schema_name or 'default'}")
        
        # Get schema analysis (inline implementation)
        try:
            tables = db_manager.get_tables(schema_name)
            logger.debug(f"Found {len(tables)} tables in schema '{schema_name or 'default'}'")
            
            # Sequential table analysis 
            all_table_info = []
            for table_name in tables:
                try:
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
                            "row_count": table_info.row_count,
                            "sample_data": table_info.sample_data
                        }
                        all_table_info.append(table_dict)
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table_name}: {e}")
            
            schema_data = {
                "schema": schema_name or "default",
                "table_count": len(all_table_info),
                "tables": all_table_info
            }
            
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        # Get relationships (inline implementation)
        relationships = {}
        table_names = [table['name'] for table in all_table_info]
        
        for table_name in table_names:
            table_data = next((t for t in all_table_info if t['name'] == table_name), None)
            if table_data and table_data.get('foreign_keys'):
                relationships[table_name] = table_data['foreign_keys']
            
        result = {
            "schema_analysis": schema_data,
            "relationships": relationships,
            "ontology": None,
            "sql_hints": {
                "workflow": [
                    "1. Review the schema_analysis to understand table structure",
                    "2. Use the ontology for business context and SQL references",
                    "3. Check relationships for potential fan-traps before JOINs",
                    "4. Validate SQL syntax before execution",
                    "5. Execute queries with appropriate limits"
                ]
            }
        }
        
        # Generate ontology if requested (inline implementation)
        if include_ontology:
            try:
                # Convert schema data to TableInfo objects for ontology generation
                tables_info = []
                for table_dict in all_table_info:
                    columns = []
                    for col_dict in table_dict['columns']:
                        col_info = ColumnInfo(
                            name=col_dict['name'],
                            data_type=col_dict['data_type'],
                            is_nullable=col_dict['is_nullable'],
                            is_primary_key=col_dict['is_primary_key'],
                            is_foreign_key=col_dict['is_foreign_key'],
                            foreign_key_table=col_dict['foreign_key_table'],
                            foreign_key_column=col_dict['foreign_key_column'],
                            comment=col_dict['comment']
                        )
                        columns.append(col_info)
                    
                    table_info = TableInfo(
                        name=table_dict['name'],
                        schema=table_dict['schema'],
                        columns=columns,
                        primary_keys=table_dict['primary_keys'],
                        foreign_keys=table_dict['foreign_keys'],
                        comment=table_dict['comment'],
                        row_count=table_dict['row_count'],
                        sample_data=table_dict['sample_data']
                    )
                    tables_info.append(table_info)
                
                # Generate ontology
                uri = config.ontology_base_uri
                generator = OntologyGenerator(base_uri=uri)
                ontology_ttl = generator.generate_from_schema(tables_info)
                
                if ontology_ttl:
                    result["ontology"] = ontology_ttl
                    result["sql_hints"]["ontology_benefits"] = [
                        "Contains ready-to-use SQL column references (e.g., customers.customer_id)",
                        "Includes complete JOIN conditions for relationships", 
                        "Provides business descriptions for understanding data meaning",
                        "Shows data types, constraints, and row counts",
                        "Acts as both documentation and SQL generation reference"
                    ]
                else:
                    logger.warning("Failed to generate ontology for analysis context")
            except Exception as e:
                logger.warning(f"Could not generate ontology for analysis context: {e}")
        
        # Add relationship warnings for analysis
        fan_trap_warnings = []
        for table, fks in relationships.items():
            if len(fks) > 1:
                referenced_tables = [fk['referenced_table'] for fk in fks]
                fan_trap_warnings.append({
                    "table": table,
                    "warning": f"Table {table} connects to multiple tables - potential fan-trap risk",
                    "referenced_tables": referenced_tables,
                    "recommendation": "Use separate CTEs or UNION approach for multi-fact aggregations"
                })
        
        if fan_trap_warnings:
            result["sql_hints"]["fan_trap_warnings"] = fan_trap_warnings
            
        logger.info(f"Generated analysis context: {len(schema_data.get('tables', []))} tables, "
                   f"ontology: {result['ontology'] is not None}")
                   
        return result
        
    except Exception as e:
        logger.error(f"Error generating analysis context: {e}")
        return create_error_response(
            f"Failed to generate analysis context: {str(e)}",
            "internal_error"
        )

@mcp.tool()
def generate_ontology(
    schema_name: Optional[str] = None,
    base_uri: Optional[str] = None,
    enrich_llm: bool = False
) -> str:
    """Generate a database ontology with direct SQL generation support.
    
    ℹ️  Most users should use get_analysis_context() instead, which includes ontology automatically.
    
    This tool generates a comprehensive ontology containing:
    - Direct database table/column references (customers.customer_id)
    - Ready-to-use JOIN conditions (orders.customer_id = customers.customer_id)
    - Business-friendly descriptions for understanding data meaning
    - Complete metadata (data types, constraints, row counts)
    
    Args:
        schema_name: Name of the schema to generate ontology from (optional)
        base_uri: Base URI for the ontology (optional, uses config default)
        enrich_llm: Whether to enrich the ontology with LLM insights (default: False)
    
    Returns:
        RDF ontology in Turtle format with complete database mappings
    """
    try:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Please use connect_database tool first to establish a connection"
            )
        
        try:
            tables = db_manager.get_tables(schema_name)
            if not tables:
                return create_error_response(
                    f"No tables found in schema '{schema_name or 'default'}'",
                    "data_error"
                )
            
            # Analyze tables
            tables_info = []
            for table_name in tables:
                try:
                    table_info = db_manager.analyze_table(table_name, schema_name)
                    if table_info:
                        tables_info.append(table_info)
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table_name}: {e}")
            
            if not tables_info:
                return create_error_response(
                    f"Could not analyze any tables in schema '{schema_name or 'default'}'",
                    "data_error"
                )
            
            # Generate ontology
            uri = base_uri or config.ontology_base_uri
            generator = OntologyGenerator(base_uri=uri)
            ontology_ttl = generator.generate_from_schema(tables_info)
            
            logger.info(f"Generated ontology for schema '{schema_name or 'default'}': {len(tables_info)} tables")
            return ontology_ttl
            
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
    except Exception as e:
        logger.error(f"Error generating ontology: {e}")
        return create_error_response(
            f"Failed to generate ontology: {str(e)}",
            "internal_error"
        )

@mcp.tool()
def sample_table_data(
    table_name: str,
    schema_name: Optional[str] = None,
    limit: int = 10
) -> Union[List[Dict[str, Any]], str]:
    """Sample data from a specific table for exploration and analysis.
    
    Args:
        table_name: Name of the table to sample
        schema_name: Name of the schema containing the table (optional)
        limit: Maximum number of rows to return (default: 10, max: 1000)
    
    Returns:
        List of dictionaries representing sample rows or error response
    """
    with error_handler("sample_table_data") as handler:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        try:
            sample_data = db_manager.sample_table_data(table_name, schema_name, limit)
            logger.info(f"Retrieved {len(sample_data)} sample rows from {table_name}")
            return sample_data
        except Exception as e:
            logger.error(f"Failed to sample table data: {e}")
            return create_error_response(
                f"Failed to sample data from table '{table_name}'",
                "database_error",
                str(e)
            )

@mcp.tool()
def validate_sql_syntax(sql_query: str) -> Dict[str, Any]:
    """Validate SQL query syntax using database-level validation.
    
    Uses the database's own SQL parser to provide accurate syntax validation
    and meaningful error messages for query correction.
    
    Args:
        sql_query: SQL query to validate
        
    Returns:
        Dictionary with validation results including any errors and suggestions
    """
    try:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        validation_result = db_manager.validate_sql_syntax(sql_query)
        logger.info(f"SQL validation completed: {'valid' if validation_result['is_valid'] else 'invalid'}")
        return validation_result
        
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"SQL validation error: {str(e)}",
            "error_type": "validation_error"
        }

@mcp.tool()
def execute_sql_query(
    sql_query: str, 
    limit: int = 1000
) -> Union[Dict[str, Any], str]:
    """Execute a validated SQL query and return results safely.
    
    ## 🚀 **STREAMLINED WORKFLOW** (Only 4 Steps):

    1. **Connect**: Use `connect_database()` to establish connection
    2. **Analyze**: Use `get_analysis_context()` - gets schema + ontology + relationships automatically
    3. **Validate**: Use `validate_sql_syntax()` before execution  
    4. **Execute**: Use `execute_sql_query()` to run validated queries

    ## 🎯 **Using the Ontology for Accurate SQL**:
    The `get_analysis_context()` tool provides an ontology containing:
    - **Ready-to-use SQL column references**: `customers.customer_id`, `orders.order_total`
    - **Complete JOIN conditions**: `orders.customer_id = customers.customer_id`
    - **Business context**: "Customer information and profile data"
    
    Extract these from the ontology TTL format and use them directly in your SQL queries.
    
    Args:
        sql_query: SQL query to execute (must pass validation first)
        limit: Maximum number of rows to return (default: 1000, max: 5000)
        
    Returns:
        Dictionary with query results and execution metadata
    """
    try:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established", 
                "connection_error",
                "Use connect_database tool first"
            )
        
        result = db_manager.execute_sql_query(sql_query, limit)
        if result['success']:
            logger.info(f"SQL query executed successfully: {result.get('row_count', 0)} rows returned")
        else:
            logger.warning(f"SQL query failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return create_error_response(
            f"Failed to execute SQL query: {str(e)}",
            "execution_error"
        )

@mcp.tool()
def get_server_info() -> Dict[str, Any]:
    """Get information about the streamlined MCP server and its capabilities.
    
    Returns:
        Dictionary containing server information and available tools
    """
    return {
        "name": "Streamlined Database Ontology MCP Server",
        "version": "2.0.0", 
        "description": "Focused MCP server with 8 essential tools for database analysis with automatic ontology generation",
        "supported_databases": ["PostgreSQL", "Snowflake"],
        "features": [
            "🌟 Single main analysis tool with automatic ontology generation",
            "🚀 Streamlined 4-step workflow (Connect → Analyze → Validate → Execute)",
            "🎯 Self-sufficient ontologies with direct SQL references",
            "🔧 Ready-to-use JOIN conditions and business context",
            "⚡ Enhanced performance with consolidated functionality",
            "🎨 Clean, focused interface with 8 essential tools"
        ],
        "tools": [
            "connect_database",        # Connect to PostgreSQL or Snowflake database
            "list_schemas",           # List available database schemas  
            "get_analysis_context",   # 🚀 MAIN TOOL: Complete analysis with automatic ontology
            "generate_ontology",      # Generate ontology manually (if needed)
            "sample_table_data",      # Sample data from specific tables
            "validate_sql_syntax",    # Validate SQL before execution
            "execute_sql_query",      # Execute validated SQL queries
            "get_server_info"         # Server information and capabilities
        ],
        "workflow": {
            "recommended_steps": [
                "1. connect_database() - Connect to your database",
                "2. get_analysis_context() - Get complete schema + ontology + relationships", 
                "3. validate_sql_syntax() - Validate your SQL queries",
                "4. execute_sql_query() - Execute validated queries"
            ],
            "main_tool": "get_analysis_context",
            "main_tool_benefits": [
                "Automatic ontology generation with SQL references",
                "Complete schema analysis in one call",
                "Relationship mapping and fan-trap warnings", 
                "Business context for data understanding"
            ]
        },
        "configuration": {
            "log_level": config.log_level,
            "ontology_base_uri": config.ontology_base_uri,
            "max_query_limit": 5000,
            "supported_formats": ["turtle"]
        }
    }

# =============================================================================
# Server Cleanup
# =============================================================================

def cleanup_server():
    """Clean up server resources on shutdown."""
    global _db_manager
    if _db_manager and _db_manager.engine:
        _db_manager.disconnect()
        logger.info("Database connections closed")

if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        cleanup_server()