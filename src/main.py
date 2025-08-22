"""Main MCP server application using FastMCP with enhanced error handling and security."""

import json
import logging
import sys
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError

from .config import config_manager
from .constants import (
    SUPPORTED_DB_TYPES, 
    DEFAULT_SAMPLE_LIMIT, 
    MAX_ENRICHMENT_SAMPLES,
    MAX_SAMPLE_LIMIT
)
from .database_manager import DatabaseManager, TableInfo
from .ontology_generator import OntologyGenerator
from .utils import setup_logging, sanitize_for_logging, sanitize_sql_for_logging
from .connection_cache import save_connection_params, load_connection_params

# Initialize logging
config = config_manager.get_server_config()
logger = setup_logging(config.log_level)

# --- MCP Server Setup ---
mcp = FastMCP("Database Ontology MCP Server")

# --- Simple Global Connection Cache ---

# Global database manager - simple and direct
_db_manager: Optional[DatabaseManager] = None
_thread_pool: Optional[ThreadPoolExecutor] = None

def get_db_manager() -> DatabaseManager:
    """Get or create global database manager with persistent auto-reconnection."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        logger.info("Created global DatabaseManager")
    
    # Auto-reconnect using persistent cache if no connection
    logger.debug(f"get_db_manager: Engine exists: {_db_manager.has_engine()}")
    if not _db_manager.has_engine():
        logger.info("get_db_manager: No engine found, attempting reconnection")
        
        # Try database manager's stored params first
        if _db_manager._last_connection_params:
            logger.info("Auto-reconnecting using DatabaseManager stored parameters")
            _db_manager.restore_connection_if_needed()
        else:
            logger.info("No DatabaseManager stored params, trying persistent cache")
            # Fall back to persistent cache
            cached_params = load_connection_params()
            if cached_params:
                logger.info(f"Found cached params for {cached_params.get('type', 'unknown')} database")
                try:
                    if cached_params["type"] == "postgresql":
                        logger.info("Attempting PostgreSQL reconnection from cache")
                        success = _db_manager.connect_postgresql(
                            cached_params["host"], cached_params["port"], cached_params["database"],
                            cached_params["username"], cached_params["password"]
                        )
                    else:  # snowflake
                        logger.info("Attempting Snowflake reconnection from cache")
                        success = _db_manager.connect_snowflake(
                            cached_params["account"], cached_params["username"], cached_params["password"],
                            cached_params["warehouse"], cached_params["database"], cached_params.get("schema", "PUBLIC")
                        )
                    
                    if success:
                        logger.info("✅ Successfully restored connection from persistent cache")
                    else:
                        logger.error("❌ Failed to restore connection from persistent cache")
                except Exception as e:
                    logger.error(f"❌ Error restoring connection from cache: {e}")
            else:
                logger.warning("No cached connection parameters found")
    else:
        logger.debug("get_db_manager: Engine already exists, connection OK")
    
    logger.info(f"get_db_manager: Final state - Engine: {_db_manager.has_engine()}")
    return _db_manager

def get_thread_pool() -> ThreadPoolExecutor:
    """Get or create global thread pool, handling shutdown state."""
    global _thread_pool
    if _thread_pool is None or _thread_pool._shutdown:
        if _thread_pool is not None:
            logger.debug("Thread pool was shut down, creating new one")
        _thread_pool = ThreadPoolExecutor(max_workers=4)
        logger.debug("Created new ThreadPoolExecutor")
    return _thread_pool

# --- Enhanced Error Response Models ---

class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: str
    error_type: str = "unknown"
    details: Optional[str] = None
    timestamp: Optional[str] = None
    request_id: Optional[str] = None

class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field details."""
    error_type: str = "validation_error"
    field_errors: Optional[Dict[str, List[str]]] = None

class ConnectionErrorResponse(ErrorResponse):
    """Connection error response."""
    error_type: str = "connection_error"
    retry_after: Optional[int] = None

def create_error_response(
    error_msg: str, 
    error_type: str = "unknown", 
    details: Optional[str] = None,
    **kwargs
) -> str:
    """Create a standardized error response."""
    response = ErrorResponse(
        error=error_msg, 
        error_type=error_type, 
        details=details,
        **kwargs
    )
    return response.model_dump_json()

@contextmanager
def error_handler(operation_name: str):
    """Context manager for consistent error handling."""
    try:
        yield
    except ValidationError as e:
        error_msg = f"Validation error in {operation_name}"
        logger.error(f"{error_msg}: {e}")
        return create_error_response(
            error_msg,
            "validation_error",
            str(e)
        )
    except ValueError as e:
        error_msg = f"Invalid input in {operation_name}"
        logger.error(f"{error_msg}: {e}")
        return create_error_response(
            error_msg,
            "validation_error",
            str(e)
        )
    except RuntimeError as e:
        error_msg = f"Runtime error in {operation_name}"
        logger.error(f"{error_msg}: {e}")
        return create_error_response(
            error_msg,
            "runtime_error",
            str(e)
        )
    except ConnectionError as e:
        error_msg = f"Connection error in {operation_name}"
        logger.error(f"{error_msg}: {e}")
        return create_error_response(
            error_msg,
            "connection_error",
            str(e)
        )
    except Exception as e:
        error_msg = f"Unexpected error in {operation_name}"
        logger.error(f"{error_msg}: {type(e).__name__}: {e}")
        return create_error_response(
            error_msg,
            "internal_error",
            f"{type(e).__name__}: {str(e)}"
        )

# --- MCP Prompts ---

@mcp.prompt()
def ontology_enrichment_guide() -> str:
    """Provides guidance for enriching database ontologies with meaningful names and descriptions."""
    return """You are an expert in ontology engineering and database schema analysis. 

Your task is to analyze database schema information and provide enrichment suggestions to make the ontology more meaningful and semantically rich.

When analyzing database schemas, consider:

1. **Business Domain Context**: Look at table and column names to understand the business domain
2. **Data Relationships**: Analyze foreign key relationships to understand entity connections  
3. **Data Types and Constraints**: Use column types, nullability, and keys to infer semantic meaning
4. **Sample Data**: Examine actual data values to better understand the purpose of each field

For ontology enrichment, provide suggestions in this exact JSON format:

```json
{
  "classes": [
    {
      "original_name": "table_name",
      "suggested_name": "MeaningfulClassName", 
      "description": "Clear description of what this entity represents"
    }
  ],
  "properties": [
    {
      "table_name": "table_name",
      "original_name": "column_name",
      "suggested_name": "meaningfulPropertyName",
      "description": "Clear description of what this property represents"
    }
  ],
  "relationships": [
    {
      "from_table": "source_table",
      "to_table": "target_table", 
      "suggested_name": "meaningfulRelationshipName",
      "description": "Clear description of what this relationship represents"
    }
  ]
}
```

**Naming Guidelines:**
- Class names: PascalCase (e.g., CustomerOrder, ProductCategory)
- Property names: camelCase (e.g., firstName, createdDateTime)
- Relationship names: camelCase (e.g., belongsToCustomer, hasOrderItems)
- Use domain-specific terminology when appropriate
- Avoid abbreviations unless they're standard in the domain

**Description Guidelines:**
- Be specific about the business purpose
- Explain constraints and business rules when evident
- Mention cardinality and optionality implications
- Reference related entities to provide context"""

@mcp.prompt()
def ontology_analysis_prompt(schema_data: Dict[str, Any]) -> str:
    """Analyzes database schema data to provide ontology enrichment suggestions."""
    return f"""Please analyze the following database schema and provide ontology enrichment suggestions.

**Database Schema Information:**
{json.dumps(schema_data.get('schema_data', []), indent=2, default=str)}

**Instructions:**
{schema_data.get('instructions', {}).get('task', 'Analyze and enrich the ontology')}

**Expected Response Format:**
The response must be a valid JSON object with exactly these three keys: "classes", "properties", and "relationships". Each should contain arrays of suggestion objects as shown in the example format.

**Guidelines:**
{chr(10).join('- ' + guideline for guideline in schema_data.get('instructions', {}).get('guidelines', []))}

Analyze the schema carefully, considering table names, column types, relationships, and any sample data provided. Focus on the most important and commonly-used entities first.

Provide meaningful, business-oriented names and descriptions that would make sense to domain experts."""

@mcp.prompt()
def sql_generation_context_prompt(enriched_context: Dict[str, Any]) -> str:
    """Provides enriched database context for improved SQL generation from business language."""
    
    schema_info = enriched_context.get('schema_info', {})
    semantic_mappings = enriched_context.get('semantic_mappings', {})
    sql_hints = enriched_context.get('sql_generation_hints', {})
    relationships = enriched_context.get('relationships', {})
    
    # Create a comprehensive context for SQL generation
    context_parts = []
    
    context_parts.append("# Database Schema Context for SQL Generation")
    context_parts.append("")
    context_parts.append("You are helping generate SQL queries from business language using this database schema.")
    context_parts.append("This context combines technical schema information with semantic business meanings.")
    context_parts.append("")
    
    # Add schema overview
    context_parts.append("## Schema Overview")
    context_parts.append(f"- Database Schema: {enriched_context.get('metadata', {}).get('schema_name', 'default')}")
    context_parts.append(f"- Total Tables: {enriched_context.get('metadata', {}).get('table_count', 0)}")
    context_parts.append("")
    
    # Add table information with semantic mappings
    context_parts.append("## Tables and Business Meanings")
    context_parts.append("")
    
    for table in schema_info.get('tables', []):
        table_name = table['name']
        semantic_info = semantic_mappings.get(table_name, {})
        
        context_parts.append(f"### {table_name}")
        context_parts.append(f"- **Business Name**: {semantic_info.get('business_name', table_name.replace('_', ' ').title())}")
        default_desc = f"Represents {table_name.replace('_', ' ').lower()} data"
        context_parts.append(f"- **Description**: {semantic_info.get('description', default_desc)}")
        context_parts.append(f"- **Row Count**: {table.get('row_count', 'Unknown')}")
        
        # Add column information
        context_parts.append("- **Columns**:")
        for col in table.get('columns', []):
            col_name = col['name']
            col_semantic = semantic_info.get('columns', {}).get(col_name, {})
            
            col_desc = col_semantic.get('description', col_name.replace('_', ' ').title())
            key_info = ""
            if col.get('is_primary_key'):
                key_info = " (PRIMARY KEY)"
            elif col.get('is_foreign_key'):
                key_info = f" (FOREIGN KEY → {col.get('foreign_key_table', '')}.{col.get('foreign_key_column', '')})"
            
            context_parts.append(f"  - `{col_name}` ({col['data_type']}){key_info}: {col_desc}")
        
        context_parts.append("")
    
    # Add relationship information
    if relationships:
        context_parts.append("## Table Relationships")
        context_parts.append("")
        for table, fks in relationships.items():
            if fks:
                context_parts.append(f"**{table}** connects to:")
                for fk in fks:
                    context_parts.append(f"- {fk['referenced_table']} via {table}.{fk['column']} = {fk['referenced_table']}.{fk['referenced_column']}")
                context_parts.append("")
    
    # Add SQL generation warnings and hints
    if sql_hints.get('relationship_warnings'):
        context_parts.append("## ⚠️ SQL Generation Warnings")
        context_parts.append("")
        for warning in sql_hints['relationship_warnings']:
            context_parts.append(f"- **{warning['table']}**: {warning['warning']}")
            context_parts.append(f"  - Referenced tables: {', '.join(warning['referenced_tables'])}")
            context_parts.append(f"  - **Recommendation**: {warning['recommendation']}")
            context_parts.append("")
    
    # Add join recommendations
    if sql_hints.get('join_recommendations'):
        context_parts.append("## SQL Join Patterns")
        context_parts.append("")
        context_parts.append("**Safe Join Conditions:**")
        for join in sql_hints['join_recommendations']:
            context_parts.append(f"- {join['from_table']} → {join['to_table']}: `{join['join_condition']}`")
        context_parts.append("")
    
    # Add SQL best practices based on the schema
    context_parts.append("## SQL Generation Best Practices for This Schema")
    context_parts.append("")
    context_parts.append("1. **Always validate syntax** using `validate_sql_syntax` before executing")
    context_parts.append("2. **Check for fan-traps** when joining multiple tables with aggregations")
    context_parts.append("3. **Use table aliases** for readability: `customers c`, `orders o`")
    context_parts.append("4. **Prefer explicit JOIN syntax** over comma-separated tables")
    context_parts.append("5. **Use LIMIT clauses** for exploratory queries")
    context_parts.append("")
    
    if sql_hints.get('relationship_warnings'):
        context_parts.append("6. **For multi-fact queries**: Use UNION approach or separate CTEs to avoid data multiplication")
        context_parts.append("7. **When aggregating**: Ensure you're not accidentally multiplying facts through joins")
        context_parts.append("")
    
    context_parts.append("## Next Steps")
    context_parts.append("")
    context_parts.append("1. Use this context to understand business intent behind user queries")
    context_parts.append("2. Translate business terms to appropriate table/column names")
    context_parts.append("3. Apply relationship knowledge for accurate joins")
    context_parts.append("4. Validate SQL syntax before execution")
    context_parts.append("5. Execute queries safely with appropriate limits")
    
    return '\n'.join(context_parts)

# --- Enhanced MCP Tools with Better Validation ---

@mcp.tool()
def connect_database(db_type: str) -> str:
    """Connect to a database using predefined credentials from environment configuration.
    
    This tool automatically uses the database credentials configured in the .env file,
    eliminating the need for users to provide connection details in chat.
    
    Args:
        db_type: Database type - must be 'postgresql' or 'snowflake'
    
    Returns:
        Success message or error JSON
    """
    with error_handler("connect_database") as handler:
        # Validate database type
        if not db_type or db_type not in SUPPORTED_DB_TYPES:
            return create_error_response(
                f"Invalid database type '{db_type}'. Supported types: {', '.join(SUPPORTED_DB_TYPES)}",
                "validation_error"
            )
        
        db_manager = get_db_manager()
        logger.info(f"connect_database: DatabaseManager ID: {id(db_manager)}")
        
        # Get database configuration from environment
        try:
            config_validation = config_manager.validate_db_config(db_type)
            if not config_validation["valid"]:
                missing_params = config_validation["missing_params"]
                return create_error_response(
                    f"Missing required environment configuration for {db_type}: {', '.join(missing_params)}",
                    "configuration_error",
                    f"Please set the following environment variables in .env: {', '.join([p.upper() for p in missing_params])}"
                )
            
            # Get the validated configuration
            db_config = config_manager.get_database_config()
            config_params = config_validation["config"]
            
        except ValueError as e:
            return create_error_response(str(e), "validation_error")
        
        # Attempt connection using predefined credentials
        try:
            if db_type == "postgresql":
                success = db_manager.connect_postgresql(
                    host=db_config.postgres_host,
                    port=db_config.postgres_port,
                    database=db_config.postgres_database,
                    username=db_config.postgres_username,
                    password=db_config.postgres_password
                )
                connection_info = {
                    "database": db_config.postgres_database,
                    "host": db_config.postgres_host,
                    "port": db_config.postgres_port
                }
            else:  # snowflake
                success = db_manager.connect_snowflake(
                    account=db_config.snowflake_account,
                    username=db_config.snowflake_username,
                    password=db_config.snowflake_password,
                    warehouse=db_config.snowflake_warehouse,
                    database=db_config.snowflake_database,
                    schema=db_config.snowflake_schema
                )
                connection_info = {
                    "database": db_config.snowflake_database,
                    "account": db_config.snowflake_account,
                    "warehouse": db_config.snowflake_warehouse,
                    "schema": db_config.snowflake_schema
                }
            
            if success:
                # Save connection parameters to persistent cache
                if db_manager._last_connection_params:
                    save_connection_params(db_manager._last_connection_params)
                
                # Debug connection state after successful connection
                logger.info(f"connect_database: Connection successful - Engine: {db_manager.has_engine()}, Params stored: {db_manager._last_connection_params is not None}")
                
                # Create safe logging info (without passwords)
                safe_info = sanitize_for_logging({
                    "db_type": db_type,
                    **{k: v for k, v in connection_info.items() if k != "password"}
                })
                logger.info(f"Successfully connected to {db_type}: {safe_info}")
                
                # Create user-friendly success message
                if db_type == "postgresql":
                    return f"✅ Successfully connected to PostgreSQL database: {connection_info['database']} at {connection_info['host']}:{connection_info['port']}"
                else:
                    return f"✅ Successfully connected to Snowflake database: {connection_info['database']} (account: {connection_info['account']}, warehouse: {connection_info['warehouse']})"
            else:
                return create_error_response(
                    f"Failed to connect to {db_type} database using configured credentials",
                    "connection_error",
                    "Check your .env file configuration and network connectivity"
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
        
        # Ensure connection is available
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error", 
                "Use connect_database tool first"
            )
        
        try:
            schemas = db_manager.get_schemas()
            logger.debug(f"Retrieved {len(schemas)} schemas")
            return schemas if schemas else []
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )

@mcp.tool()
def analyze_schema(schema_name: Optional[str] = None) -> Union[Dict[str, Any], str]:
    """Analyze a database schema and return detailed table information.
    
    Args:
        schema_name: Name of the schema to analyze (optional)
    
    Returns:
        Dictionary containing tables and their detailed information or error response
    """
    with error_handler("analyze_schema") as handler:
        db_manager = get_db_manager()
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
            
            result = {
                "schema": schema_name or "default",
                "table_count": len(all_table_info),
                "tables": all_table_info
            }
            
            logger.info(f"Successfully analyzed schema '{schema_name or 'default'}': {len(all_table_info)} tables")
            return result
            
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )

@mcp.tool()
def generate_ontology(
    schema_name: Optional[str] = None,
    base_uri: Optional[str] = None,
    enrich_llm: bool = False
) -> str:
    """Generate an RDF ontology from the database schema.
    
    Args:
        schema_name: Name of the schema to generate ontology from (optional)
        base_uri: Base URI for the ontology (optional, uses config default)
        enrich_llm: Whether to enrich the ontology with LLM insights (default: False)
    
    Returns:
        RDF ontology in Turtle format or error response
    """
    try:
        db_manager = get_db_manager()
        
        # Debug connection state
        logger.info(f"generate_ontology: DatabaseManager ID: {id(db_manager)}")
        logger.info(f"generate_ontology: Has engine: {db_manager.has_engine()}")
        logger.info(f"generate_ontology: Has stored params: {db_manager._last_connection_params is not None}")
        
        # Connection check with clear error message
        if not db_manager.has_engine():
            logger.error("generate_ontology: No database engine - connection was lost or never established")
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Please use connect_database tool first to establish a connection"
            )
        
        tables = db_manager.get_tables(schema_name)
        if not tables:
            return create_error_response(
                f"No tables found in schema '{schema_name or 'default'}'",
                "data_error",
                "Schema may not exist or may be empty"
            )
        
        # Sequential table analysis (simplified to avoid thread pool issues)
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
        
        if enrich_llm:
            # Sample data for enrichment
            data_samples = {}
            for table in tables_info[:10]:  # Limit to first 10 tables for performance
                try:
                    samples = db_manager.sample_table_data(
                        table.name, 
                        schema_name, 
                        limit=MAX_ENRICHMENT_SAMPLES
                    )
                    if samples:
                        data_samples[table.name] = samples
                except Exception as e:
                    logger.warning(f"Could not sample data from table {table.name}: {e}")
            
            try:
                ontology_ttl = generator.enrich_with_llm(tables_info, data_samples)
            except Exception as e:
                logger.warning(f"LLM enrichment failed, using basic ontology: {e}")
            
        logger.info(f"Generated ontology for schema '{schema_name or 'default'}': {len(tables_info)} tables")
        return ontology_ttl
        
    except Exception as e:
        logger.error(f"Error in generate_ontology: {e}")
        return create_error_response(
            f"Failed to generate ontology: {str(e)}",
            "internal_error"
        )

@mcp.tool()
def sample_table_data(
    table_name: str,
    schema_name: Optional[str] = None,
    limit: int = DEFAULT_SAMPLE_LIMIT
) -> Union[List[Dict[str, Any]], str]:
    """Sample data from a specific table for analysis.
    
    Args:
        table_name: Name of the table to sample (required)
        schema_name: Schema containing the table (optional)
        limit: Maximum number of rows to return (default: 10, max: 1000)
    
    Returns:
        List of sample rows as dictionaries or error response
    """
    with error_handler("sample_table_data") as handler:
        if not table_name:
            return create_error_response(
                "Table name is required",
                "validation_error"
            )
        
        db_manager = get_db_manager()
        try:
            sample_data = db_manager.sample_table_data(table_name, schema_name, limit)
            logger.debug(f"Sampled {len(sample_data)} rows from table {table_name}")
            return sample_data
        except (ValueError, RuntimeError) as e:
            return create_error_response(str(e), "validation_error")

@mcp.tool()
def get_table_relationships(schema_name: Optional[str] = None) -> Union[Dict[str, List[Dict[str, str]]], str]:
    """Get foreign key relationships between tables in a schema.
    
    Args:
        schema_name: Name of the schema to analyze relationships (optional)
    
    Returns:
        Dictionary mapping table names to their foreign key relationships or error response
    """
    with error_handler("get_table_relationships") as handler:
        db_manager = get_db_manager()
        try:
            relationships = db_manager.get_table_relationships(schema_name)
            logger.debug(f"Retrieved relationships for {len(relationships)} tables")
            return relationships
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )

@mcp.tool()
def get_enrichment_data(schema_name: Optional[str] = None) -> Union[Dict[str, Any], str]:
    """Get structured data for ontology enrichment analysis.
    
    Args:
        schema_name: Name of the schema to get enrichment data for (optional)
    
    Returns:
        Dictionary containing schema data and enrichment instructions or error response
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
        
        tables = db_manager.get_tables(schema_name)
        if not tables:
            return create_error_response(
                f"No tables found in schema '{schema_name or 'default'}'",
                "data_error"
            )
        
        # Analyze tables in parallel
        tables_info = []
        with get_thread_pool() as executor:
            future_to_table = {
                executor.submit(db_manager.analyze_table, table_name, schema_name): table_name
                for table_name in tables
            }
            
            for future in as_completed(future_to_table):
                try:
                    table_info = future.result()
                    if table_info:
                        tables_info.append(table_info)
                except Exception as e:
                    logger.warning(f"Failed to analyze table: {e}")
        
        if not tables_info:
            return create_error_response(
                f"Could not analyze any tables in schema '{schema_name or 'default'}'",
                "data_error"
            )
        
        # Get sample data for enrichment
        data_samples = {}
        for table in tables_info[:10]:  # Limit for performance
            try:
                samples = db_manager.sample_table_data(
                    table.name, 
                    schema_name, 
                    limit=MAX_ENRICHMENT_SAMPLES
                )
                if samples:
                    data_samples[table.name] = samples
            except Exception as e:
                logger.warning(f"Could not sample data from table {table.name}: {e}")
        
        uri = config.ontology_base_uri
        generator = OntologyGenerator(base_uri=uri)
        enrichment_data = generator.get_enrichment_data(tables_info, data_samples)
        
        logger.info(f"Generated enrichment data for {len(tables_info)} tables")
        return enrichment_data
        
    except Exception as e:
        logger.error(f"Error in get_enrichment_data: {e}")
        return create_error_response(
            f"Failed to get enrichment data: {str(e)}",
            "internal_error"
        )

@mcp.tool()
def apply_ontology_enrichment(
    schema_name: Optional[str] = None,
    base_uri: Optional[str] = None,
    enrichment_suggestions: Optional[Dict[str, Any]] = None
) -> str:
    """Apply enrichment suggestions to generate an enriched ontology.
    
    Args:
        schema_name: Name of the schema to generate ontology from (optional)
        base_uri: Base URI for the ontology (optional)
        enrichment_suggestions: Dictionary containing enrichment suggestions
    
    Returns:
        Enriched RDF ontology in Turtle format or error response
    """
    with error_handler("apply_ontology_enrichment") as handler:
        if not enrichment_suggestions:
            return create_error_response(
                "enrichment_suggestions parameter is required",
                "validation_error"
            )
        
        # Validate enrichment suggestions structure
        required_keys = {"classes", "properties", "relationships"}
        if not all(key in enrichment_suggestions for key in required_keys):
            missing = required_keys - set(enrichment_suggestions.keys())
            return create_error_response(
                f"Missing required keys in enrichment_suggestions: {missing}",
                "validation_error"
            )
        
        db_manager = get_db_manager()
        
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
            
            # Generate and enrich ontology
            uri = base_uri or config.ontology_base_uri
            generator = OntologyGenerator(base_uri=uri)
            base_ontology = generator.generate_from_schema(tables_info)
            
            try:
                generator.apply_enrichment(enrichment_suggestions)
                enriched_ontology = generator.serialize_ontology()
                logger.info(f"Applied enrichment to ontology with {len(enrichment_suggestions.get('classes', []))} class suggestions")
                return enriched_ontology
            except Exception as e:
                logger.error(f"Error applying enrichment: {e}")
                return create_error_response(
                    "Failed to apply enrichment suggestions",
                    "enrichment_error",
                    str(e)
                )
                
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )

@mcp.tool()
def validate_sql_syntax(sql_query: str) -> Union[Dict[str, Any], str]:
    """Validate SQL query syntax using database-level validation for LLM correction.
    
    This tool uses the database's own SQL parser (via prepared statements for PostgreSQL 
    or EXPLAIN for Snowflake) to provide accurate syntax validation and meaningful error 
    messages that LLMs can use to self-correct their generated SQL.
    
    Args:
        sql_query: SQL query to validate (text-to-SQL generated query)
    
    Returns:
        Dictionary with validation results including database-specific errors and suggestions
    """
    with error_handler("validate_sql_syntax") as handler:
        if not sql_query or not sql_query.strip():
            return create_error_response(
                "SQL query is required",
                "validation_error"
            )
        
        # Log SQL query safely (without credentials)
        safe_query = sanitize_sql_for_logging(sql_query)
        logger.info(f"Validating SQL query: {safe_query}")
        
        db_manager = get_db_manager()
        try:
            validation_result = db_manager.validate_sql_syntax(sql_query)
            
            # Log validation result
            if validation_result["is_valid"]:
                logger.info(f"SQL validation successful: {validation_result['query_type']}, affected tables: {validation_result.get('affected_tables', [])}")
            else:
                logger.warning(f"SQL validation failed: {validation_result['error_type']} - {validation_result['error']}")
            
            return validation_result
            
        except RuntimeError as e:
            return create_error_response(
                "No database connection established", 
                "connection_error",
                "Use connect_database tool first"
            )
        except Exception as e:
            return create_error_response(
                f"SQL validation error: {str(e)}",
                "validation_error"
            )

@mcp.tool()
def execute_sql_query(
    sql_query: str, 
    limit: int = 1000
) -> Union[Dict[str, Any], str]:
    """Execute a validated SQL query and return results safely.
    
    This tool executes SQL queries with built-in safety measures including automatic
    validation, result limits, and execution timeouts. Only SELECT, CTE, and metadata
    queries are allowed for security.

    Before executing an SQL query, it can be validated against the database's SQL parser 
    with the tool: validate_sql_syntax

    Additionally, the query should be checked for multi-fact scenario and rewritten
    to prevent SQL traps.

    An Ontology of the schema can be generated with the tool: generate_ontology. This 
    can be used as guidelines and guardrails for query generation.

    # Execute SQL Query Tool - Complete Documentation

    ## Overview

    Execute a validated SQL query and return results safely with built-in safety measures 
    including automatic validation, result limits, and execution timeouts. 
    Only SELECT, CTE, and metadata queries are allowed for security.

    ## Prerequisites

    Before executing any SQL query, you MUST follow this workflow:

    1. **Connect to Database**: Use `connect_database()` to establish connection
    2. **Analyze Schema**: Use `analyze_schema()` to understand table structure
    3. **Check Relationships**: Use `get_table_relationships()` to identify potential fan-traps
    4. **Generate Ontology** (optional): Use `generate_ontology()` for complex schemas
    5. **Validate Syntax**: Use `validate_sql_syntax()` before execution

    ## 🚨 CRITICAL SQL TRAP PREVENTION PROTOCOL 🚨

    ### MANDATORY PRE-EXECUTION CHECKLIST

    **1. 🔍 RELATIONSHIP ANALYSIS (REQUIRED)**
    - ALWAYS call `get_table_relationships()` first
    - Identify ALL 1:many relationships in your query
    - Flag any table appearing on "many" side of multiple relationships

    **2. 🎯 FAN-TRAP DETECTION (CRITICAL)**

    **IMMEDIATE RED FLAGS:**
    - ❌ Sales + Shipments + SUM() = GUARANTEED FAN-TRAP
    - ❌ Any fact table + dimension + aggregation = HIGH RISK
    - ❌ Multiple LEFT JOINs + GROUP BY = DANGER ZONE
    - ❌ Joining 3+ tables with SUM/COUNT/AVG = LIKELY INFLATED RESULTS

    **PATTERN CHECK:**
    ```
    If query has: FROM tableA JOIN tableB JOIN tableC 
    WHERE tableA→tableB (1:many) AND tableA→tableC (1:many)
    Then: GUARANTEED CARTESIAN PRODUCT MULTIPLICATION
    Result: SUM(tableA.amount) will be artificially inflated!
    ```

    **3. 🛡️ MANDATORY VALIDATION**
    - Call `validate_sql_syntax()` before execution
    - Review warnings about query complexity
    - Check for multiple table joins with aggregation

    ## ✅ SAFE QUERY PATTERNS

    ### 🔒 PATTERN 1 - UNION APPROACH (RECOMMENDED FOR MULTI-FACT)

    **Best for:** Multiple fact tables (sales, shipments, returns, etc.)

    ```sql
    WITH unified_facts AS (
        -- Sales facts
        SELECT 
            client_id, 
            product_id, 
            sales_amount as amount, 
            0 as shipment_qty, 
            0 as return_qty,
            'SALES' as fact_type
        FROM sales
        
        UNION ALL
        
        -- Shipment facts  
        SELECT 
            client_id, 
            product_id, 
            0 as amount, 
            shipment_quantity, 
            0 as return_qty,
            'SHIPMENT' as fact_type
        FROM shipments s JOIN sales sal ON s.sales_id = sal.id
        
        UNION ALL
        
        -- Return facts
        SELECT 
            client_id, 
            product_id, 
            0 as amount, 
            0 as shipment_qty, 
            return_quantity,
            'RETURN' as fact_type  
        FROM returns r JOIN sales sal ON r.sales_id = sal.id
    )
    SELECT 
        client_id,
        product_id,
        SUM(amount) as total_sales,
        SUM(shipment_qty) as total_shipped,
        SUM(return_qty) as total_returned
    FROM unified_facts 
    GROUP BY client_id, product_id;
    ```

    **Advantages:**
    - ✅ Natural fan-trap immunity by design
    - ✅ Unified data model for consistent aggregation
    - ✅ Easy to extend with additional fact types
    - ✅ Single aggregation logic for all measures
    - ✅ Better performance with fewer table scans

    ### 🔒 PATTERN 2 - SEPARATE AGGREGATION (LEGACY APPROACH)

    **Use when:** UNION approach is not suitable

    ```sql
    WITH fact1_totals AS (
        SELECT key, SUM(amount) as total_amount 
        FROM fact1 GROUP BY key
    ),
    fact2_totals AS (
        SELECT key, SUM(quantity) as total_quantity 
        FROM fact2 GROUP BY key
    )
    SELECT 
        f1.key, 
        f1.total_amount,
        COALESCE(f2.total_quantity, 0) as total_quantity
    FROM fact1_totals f1 
    LEFT JOIN fact2_totals f2 ON f1.key = f2.key;
    ```

    ### 🔒 PATTERN 3 - DISTINCT AGGREGATION (USE CAREFULLY)

    **Warning:** Only use when you fully understand the data relationships

    ```sql
    SELECT 
        key, 
        SUM(DISTINCT fact1.amount) as total_amount,
        SUM(fact2.quantity) as total_quantity 
    FROM fact1 
    LEFT JOIN fact2 ON fact1.id = fact2.fact1_id 
    GROUP BY key;
    ```

    ### 🔒 PATTERN 4 - WINDOW FUNCTIONS

    **For:** Complex analytical queries with preserved granularity

    ```sql
    SELECT DISTINCT 
        key, 
        SUM(amount) OVER (PARTITION BY key) as total_amount,
        pre_aggregated_quantity
    FROM fact1 
    LEFT JOIN (
        SELECT key, SUM(qty) as pre_aggregated_quantity 
        FROM fact2 GROUP BY key
    ) f2 USING(key);
    ```

    ## 🔄 RESULT VALIDATION (POST-EXECUTION)

    **Always verify results make business sense:**
    - Compare totals with business expectations
    - Verify: `SELECT SUM(amount) FROM base_table` vs your query result
    - Check row counts are reasonable
    - If results seem too high → likely fan-trap occurred

    ## 📋 COMMON DEADLY COMBINATIONS TO AVOID

    ❌ **Never do these without proper fan-trap prevention:**
    - `sales LEFT JOIN shipments + SUM(sales.amount)`
    - `orders LEFT JOIN order_items LEFT JOIN products + SUM(orders.total)`
    - `customers LEFT JOIN transactions LEFT JOIN transaction_items + aggregation`
    - Any query joining parent→child1 + parent→child2 with SUM/COUNT

    ## 🎯 RELATIONSHIP ANALYSIS EXAMPLES

    **SAFE (1:1 relationships):**
    ```
    customers → customer_profiles (1:1) ✅
    ```

    **RISKY (1:many):**
    ```
    customers → orders (1:many) ⚠️
    ```

    **DEADLY (fan-trap):**
    ```
    orders → order_items (1:many) + orders → shipments (1:many) 🚨
    ```

    **IF YOUR QUERY INCLUDES THE DEADLY PATTERN:**
    → STOP! Rewrite using UNION approach or separate aggregation CTEs

    ## 🔧 EMERGENCY FAN-TRAP FIX

    If you suspect fan-trap in existing query:
    1. **Split into UNION approach** (recommended)
    2. **Use separate aggregations**
    3. **Add DISTINCT in SUM()** as temporary fix
    4. **Validate results** against source tables
    5. **Always aggregate fact tables separately** before joining

    **Remember:** Fan-traps cause SILENT DATA CORRUPTION! Your query will execute successfully but return WRONG RESULTS. The bigger the multiplication factor, the more wrong your data becomes.

    ## ⚡ AUTOMATED CHECK

    If your query involves more than 2 tables and includes SUM/COUNT/AVG, you MUST analyze for fan-traps before execution. No exceptions!

    ## 🎯 SUCCESS CRITERIA

    Only proceed with `execute_sql_query()` after ALL checks pass:
    - [ ] Schema analyzed ✓
    - [ ] Relationships analyzed ✓  
    - [ ] Fan-trap patterns checked ✓
    - [ ] Syntax validated ✓
    - [ ] Safe aggregation pattern used ✓
    - [ ] Results make business sense ✓

    ## Function Parameters

    ### Required Parameters
    - **sql_query** (string): SQL query to execute (must be SELECT, WITH, EXPLAIN, etc.)

    ### Optional Parameters  
    - **limit** (integer): Maximum number of rows to return (default: 1000, max: 5000)

    ### Returns
    Dictionary containing:
    - **success** (boolean): Whether query executed successfully
    - **data** (array): Query results as array of objects
    - **columns** (array): Column names in result set
    - **row_count** (integer): Number of rows returned
    - **execution_time_ms** (float): Query execution time
    - **error** (string): Error message if query failed
    - **warnings** (array): Performance and complexity warnings
    - **limit_applied** (boolean): Whether result limit was applied

    ## Security Restrictions

    **ALLOWED:**
    - SELECT statements
    - Common Table Expressions (WITH)
    - EXPLAIN statements
    - Database metadata queries

    **PROHIBITED:**
    - INSERT, UPDATE, DELETE statements
    - DDL operations (CREATE, DROP, ALTER)
    - Transaction control (COMMIT, ROLLBACK)
    - System functions that modify state
    - Dynamic SQL execution

    ## Performance Guidelines

    **Query Optimization:**
    - Use appropriate indexes via schema analysis
    - Limit result sets with WHERE clauses
    - Use EXPLAIN to understand query plans
    - Monitor execution time warnings

    **Resource Management:**
    - Default limit: 1000 rows
    - Maximum limit: 5000 rows  
    - Automatic timeout protection
    - Memory usage monitoring

    ## Error Handling

    **Common Error Types:**
    - **Syntax errors**: Use `validate_sql_syntax()` first
    - **Permission errors**: Check allowed query types
    - **Timeout errors**: Simplify complex queries
    - **Memory errors**: Reduce result set size

    **Best Practices:**
    - Always validate syntax before execution
    - Start with small result sets
    - Use LIMIT clauses appropriately
    - Monitor execution time and warnings

    ## Examples

    ### Multi-Fact Query (Recommended)
    ```sql
    WITH unified_facts AS (
        SELECT customer_id, product_id, sales_amount, 0 as returns, 'SALES' as type
        FROM sales
        UNION ALL
        SELECT customer_id, product_id, 0, return_amount, 'RETURNS' as type  
        FROM returns r JOIN sales s ON r.sales_id = s.id
    )
    SELECT customer_id, SUM(sales_amount) as net_sales, SUM(returns) as total_returns
    FROM unified_facts GROUP BY customer_id;
    ```

    ### Safe Aggregation Query
    ```sql
    WITH customer_sales AS (
        SELECT customer_id, SUM(amount) as total_sales
        FROM sales GROUP BY customer_id
    )
    SELECT c.name, cs.total_sales
    FROM customers c
    LEFT JOIN customer_sales cs ON c.id = cs.customer_id
    ORDER BY cs.total_sales DESC;
    ```

    Args:
        sql_query: SQL query to execute (must be SELECT, WITH, EXPLAIN, etc.)
        limit: Maximum number of rows to return (default: 1000, max: 5000)
    
    Returns:
        Dictionary with query results, execution metadata, and any warnings
    """
    with error_handler("execute_sql_query") as handler:
        if not sql_query or not sql_query.strip():
            return create_error_response(
                "SQL query is required",
                "validation_error"
            )
        
        # Log SQL query safely (without credentials)
        safe_query = sanitize_sql_for_logging(sql_query)
        logger.info(f"Executing SQL query: {safe_query}")
        
        # Validate limit parameter
        if not isinstance(limit, int) or limit <= 0:
            limit = 1000
            logger.warning("Invalid limit parameter, using default: 1000")
        
        db_manager = get_db_manager()
        try:
            result = db_manager.execute_sql_query(sql_query, limit)
            
            # Log execution results with more detail
            if result["success"]:
                logger.info(f"SQL execution successful: {result['row_count']} rows returned in {result['execution_time_ms']}ms" +
                           (f", limit applied: {limit}" if result.get('limit_applied', False) else ""))
                if result.get('warnings'):
                    logger.info(f"SQL execution warnings: {result['warnings']}")
            else:
                logger.error(f"SQL execution failed: {result['error_type']} - {result['error']}")
            
            return result
            
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error", 
                "Use connect_database tool first"
            )
        except Exception as e:
            return create_error_response(
                f"SQL execution error: {str(e)}",
                "execution_error"
            )


@mcp.tool()
def get_enriched_schema_context(
    schema_name: Optional[str] = None,
    include_sample_data: bool = True,
    include_ontology: bool = True
) -> Union[Dict[str, Any], str]:
    """Get comprehensive schema context combining database metadata with ontology semantics.
    
    This tool provides a unified view of the database schema that combines:
    - Raw schema information (tables, columns, relationships)
    - Semantic ontology mappings (classes, properties, relationships)
    - Sample data for context
    - Business-friendly descriptions and naming suggestions
    
    This enriched context is designed to help Claude Desktop generate better SQL
    by understanding both the technical structure and semantic meaning of the data.
    
    Args:
        schema_name: Name of the schema to analyze (optional)
        include_sample_data: Whether to include sample data for context (default: True)
        include_ontology: Whether to include ontology mappings (default: True)
    
    Returns:
        Dictionary containing enriched schema context with semantic mappings
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
        
        logger.info(f"Generating enriched schema context for schema: {schema_name or 'default'}")
        
        # Step 1: Get raw schema information
        schema_data = analyze_schema(schema_name)
        if isinstance(schema_data, str):  # Error occurred
            return schema_data
            
        # Step 2: Get relationships for SQL generation context
        relationships = get_table_relationships(schema_name)
        if isinstance(relationships, str):  # Error occurred
            relationships = {}
            
        # Step 3: Generate ontology if requested
        ontology_ttl = None
        semantic_mappings = {}
        
        if include_ontology:
            try:
                # Get enrichment data for semantic understanding
                enrichment_data = get_enrichment_data(schema_name)
                if isinstance(enrichment_data, dict):
                    # Generate ontology with semantic mappings
                    ontology_ttl = generate_ontology(schema_name, enrich_llm=False)
                    
                    # Create semantic mappings for easier access
                    for table_data in enrichment_data.get('schema_data', []):
                        table_name = table_data['table_name']
                        default_table_desc = f"Represents {table_name.replace('_', ' ').lower()} data"
                        semantic_mappings[table_name] = {
                            'business_name': table_name.replace('_', ' ').title(),
                            'description': default_table_desc,
                            'columns': {}
                        }
                        
                        # Map column semantics
                        for col in table_data.get('columns', []):
                            col_name = col['name']
                            col_info = {
                                'business_name': col_name.replace('_', ' ').title(),
                                'data_type': col['data_type'],
                                'is_key': col.get('is_primary_key', False) or col.get('is_foreign_key', False),
                                'description': f"{col_name.replace('_', ' ').title()}"
                            }
                            
                            # Add more context based on column patterns
                            col_lower = col_name.lower()
                            if 'id' in col_lower:
                                col_info['description'] = f"Unique identifier for {col_name.replace('_id', '').replace('_', ' ')}"
                            elif 'name' in col_lower:
                                col_info['description'] = f"Name or title"
                            elif 'date' in col_lower or 'time' in col_lower:
                                col_info['description'] = f"Date/time information"
                            elif 'amount' in col_lower or 'price' in col_lower or 'cost' in col_lower:
                                col_info['description'] = f"Monetary or quantity value"
                            elif 'count' in col_lower or 'quantity' in col_lower:
                                col_info['description'] = f"Numeric count or quantity"
                            elif 'status' in col_lower or 'state' in col_lower:
                                col_info['description'] = f"Status or state indicator"
                                
                            semantic_mappings[table_name]['columns'][col_name] = col_info
                            
            except Exception as e:
                logger.warning(f"Failed to generate ontology context: {e}")
                
        # Step 4: Add SQL generation hints
        sql_hints = {
            'relationship_warnings': [],
            'join_recommendations': [],
            'aggregation_safe_patterns': []
        }
        
        # Analyze relationships for potential fan-traps
        for table, fks in relationships.items():
            if len(fks) > 1:
                referenced_tables = [fk['referenced_table'] for fk in fks]
                sql_hints['relationship_warnings'].append({
                    'table': table,
                    'warning': f"Table {table} has multiple foreign keys - potential fan-trap risk",
                    'referenced_tables': referenced_tables,
                    'recommendation': "Use separate CTEs or UNION approach for multi-fact queries"
                })
                
        # Add join recommendations based on relationships
        for table, fks in relationships.items():
            for fk in fks:
                sql_hints['join_recommendations'].append({
                    'from_table': table,
                    'to_table': fk['referenced_table'],
                    'join_condition': f"{table}.{fk['column']} = {fk['referenced_table']}.{fk['referenced_column']}",
                    'relationship_type': "many_to_one"
                })
                
        # Step 5: Compile enriched context
        enriched_context = {
            'schema_info': schema_data,
            'relationships': relationships,
            'semantic_mappings': semantic_mappings,
            'sql_generation_hints': sql_hints,
            'metadata': {
                'schema_name': schema_name or 'default',
                'table_count': len(schema_data.get('tables', [])),
                'has_ontology': ontology_ttl is not None,
                'has_sample_data': include_sample_data,
                'generation_timestamp': None  # Could add timestamp if needed
            }
        }
        
        # Optionally include ontology
        if ontology_ttl:
            enriched_context['ontology_ttl'] = ontology_ttl
            
        logger.info(f"Generated enriched schema context: {len(schema_data.get('tables', []))} tables, "
                   f"{len(relationships)} relationships, ontology: {ontology_ttl is not None}")
                   
        return enriched_context
        
    except Exception as e:
        logger.error(f"Error generating enriched schema context: {e}")
        return create_error_response(
            f"Failed to generate enriched schema context: {str(e)}",
            "internal_error"
        )

@mcp.tool()
def get_server_info() -> Dict[str, Any]:
    """Get information about the MCP server and its capabilities.
    
    Returns:
        Dictionary containing server information
    """
    return {
        "name": "Database Ontology MCP Server",
        "version": "0.2.0",
        "description": "Enhanced MCP server for database schema analysis and ontology generation",
        "supported_databases": SUPPORTED_DB_TYPES,
        "features": [
            "Enhanced database connection management with pooling and auto-reconnection",
            "Parallel schema analysis for improved performance", 
            "Advanced error handling and validation",
            "Connection health monitoring and recovery",
            "Structured logging and observability",
            "Security-enhanced credential handling",
            "RDF/OWL ontology generation with validation",
            "LLM-enhanced ontology enrichment",
            "Database-level SQL syntax validation for text-to-SQL",
            "Safe SQL query execution with automatic limits",
            "Comprehensive configuration management",
            "Enriched schema context combining metadata with ontology semantics"
        ],
        "tools": [
            "connect_database",
            "list_schemas", 
            "analyze_schema",
            "generate_ontology",
            "sample_table_data",
            "get_table_relationships",
            "get_enrichment_data",
            "apply_ontology_enrichment",
            "validate_sql_syntax",
            "execute_sql_query",
            "get_enriched_schema_context",
            "get_server_info"
        ],
        "configuration": {
            "log_level": config.log_level,
            "base_uri": config.ontology_base_uri,
            "max_sample_limit": MAX_SAMPLE_LIMIT,
            "supported_formats": ["turtle", "rdf/xml", "json-ld"]
        }
    }

# --- Cleanup on shutdown ---

def cleanup_server():
    """Clean up server resources."""
    try:
        global _db_manager, _thread_pool
        if _db_manager:
            _db_manager.disconnect()
            _db_manager = None
        if _thread_pool:
            _thread_pool.shutdown(wait=True)
            _thread_pool = None
        logger.info("Server cleanup completed")
    except Exception as e:
        logger.error(f"Error during server cleanup: {e}")

if __name__ == "__main__":
    try:
        logger.info("Starting Database Ontology MCP Server v0.2.0")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server startup error: {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        cleanup_server()