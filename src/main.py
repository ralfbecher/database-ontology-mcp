#!/usr/bin/env python3
"""
Streamlined Database Ontology MCP Server

A focused MCP server with 9 essential tools for database analysis with automatic ontology generation and interactive charting.
Main tool: get_analysis_context() - provides complete schema analysis with integrated ontology.
New: create_chart() - creates interactive visualizations from analytical results.
"""

import logging
import json
import asyncio
import os
import base64
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from fastmcp import FastMCP
import mcp.types as types
from .database_manager import DatabaseManager, TableInfo, ColumnInfo
from .ontology_generator import OntologyGenerator
from .config import config_manager
from . import __version__, __name__ as SERVER_NAME, __description__

# Initialize configuration and logging
server_config = config_manager.get_server_config()
logging.basicConfig(level=getattr(logging, server_config.log_level))
logger = logging.getLogger(__name__)

# Temporary directory for SVG file storage during session
import tempfile
import shutil

# Create and clean tmp directory for chart storage in project root
TMP_DIR = Path(__file__).parent.parent / "tmp"

def setup_tmp_directory():
    """Setup and clean temporary directory for chart storage."""
    try:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)
            logger.debug(f"Cleaned existing tmp directory: {TMP_DIR}")
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Chart tmp directory ready: {TMP_DIR.absolute()}")
    except Exception as e:
        logger.warning(f"Failed to setup chart tmp directory: {e}")

# Setup tmp directory on startup
setup_tmp_directory()

# Initialize FastMCP
mcp = FastMCP(SERVER_NAME)

# Chart serving no longer needed - SVG content returned directly in responses

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

# Chart storage no longer needed - SVG content returned directly


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
    schema: Optional[str] = "PUBLIC",
    role: Optional[str] = None
) -> Dict[str, Any]:
    """Connect to a PostgreSQL or Snowflake database.
    
    Parameters are optional - the tool will automatically use values from .env file when parameters are not provided.
    
    Args:
        db_type: Database type ("postgresql" or "snowflake")
        host: Database host (PostgreSQL only, uses POSTGRES_HOST from .env if not provided)
        port: Database port (PostgreSQL only, uses POSTGRES_PORT from .env if not provided) 
        database: Database name (uses POSTGRES_DATABASE or SNOWFLAKE_DATABASE from .env if not provided)
        username: Username for authentication (uses POSTGRES_USERNAME or SNOWFLAKE_USERNAME from .env if not provided)
        password: Password for authentication (uses POSTGRES_PASSWORD or SNOWFLAKE_PASSWORD from .env if not provided)
        account: Snowflake account identifier (Snowflake only, uses SNOWFLAKE_ACCOUNT from .env if not provided)
        warehouse: Snowflake warehouse (Snowflake only, uses SNOWFLAKE_WAREHOUSE from .env if not provided)
        schema: Schema name (Snowflake only, uses SNOWFLAKE_SCHEMA from .env if not provided, default: "PUBLIC")
        role: Snowflake role (Snowflake only, uses SNOWFLAKE_ROLE from .env if not provided, default: "PUBLIC")
    
    Returns:
        Connection status information or error response
        
    Examples:
        # Connect using .env file values
        connect_database("postgresql")
        connect_database("snowflake")
        
        # Override specific parameters
        connect_database("postgresql", host="custom.host.com", port=5433)
    """
    with error_handler("connect_database") as handler:
        db_manager = get_db_manager()
        db_config = config_manager.get_database_config()
        
        try:
            if db_type.lower() == "postgresql":
                # Use provided parameters or fall back to config
                final_host = host or db_config.postgres_host
                final_port = port or db_config.postgres_port
                final_database = database or db_config.postgres_database
                final_username = username or db_config.postgres_username
                final_password = password or db_config.postgres_password
                
                if not all([final_host, final_port, final_database, final_username]):
                    return create_error_response(
                        "Missing required PostgreSQL parameters: host, port, database, username (provide via parameters or .env file)",
                        "parameter_error"
                    )
                success = db_manager.connect_postgresql(final_host, final_port, final_database, final_username, final_password or "")
                
            elif db_type.lower() == "snowflake":
                # Use provided parameters or fall back to config
                final_account = account or db_config.snowflake_account
                final_username = username or db_config.snowflake_username
                final_password = password or db_config.snowflake_password
                final_warehouse = warehouse or db_config.snowflake_warehouse
                final_database = database or db_config.snowflake_database
                final_schema = schema or db_config.snowflake_schema
                final_role = role or db_config.snowflake_role
                
                if not all([final_account, final_username, final_database, final_warehouse]):
                    return create_error_response(
                        "Missing required Snowflake parameters: account, username, database, warehouse (provide via parameters or .env file)", 
                        "parameter_error"
                    )
                success = db_manager.connect_snowflake(final_account, final_username, final_password or "", final_warehouse, final_database, final_schema, final_role)
                
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
def list_schemas() -> Dict[str, Any]:
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
            return {
                "success": True,
                "schemas": schemas,
                "count": len(schemas)
            }
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
) -> Dict[str, Any]:
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
                uri = server_config.ontology_base_uri
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
            uri = base_uri or server_config.ontology_base_uri
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
) -> Dict[str, Any]:
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
            return {
                "success": True,
                "table_name": table_name,
                "schema_name": schema_name,
                "sample_data": sample_data,
                "row_count": len(sample_data),
                "limit": limit
            }
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
) -> Dict[str, Any]:
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
def generate_chart(
    data_source: List[Dict[str, Any]],
    chart_type: str,
    x_column: str,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    chart_library: str = "matplotlib",
    chart_style: str = "grouped",
    width: int = 800,
    height: int = 600
) -> list[types.ContentBlock]:
    """Generate interactive charts from SQL query result data.
    
    📊 Supports multiple chart types with both Plotly (interactive) and Matplotlib/Seaborn (static) backends.
    
    Args:
        data_source: List of dictionaries containing query results (from execute_sql_query)
        chart_type: Type of chart ("bar", "line", "scatter", "heatmap")
        x_column: Column name for X-axis
        y_column: Column name for Y-axis (required for most chart types)
        color_column: Column name for color grouping (optional)
        title: Chart title (auto-generated if not provided)
        chart_library: Library to use ("plotly" or "matplotlib")
        chart_style: Chart style ("grouped", "stacked" for bar charts)
        width: Chart width in pixels
        height: Chart height in pixels
        output_format: Output format ("image", "file") - "image" returns MCP image resource for Claude Desktop
    
    Chart Types:
        - "bar": Bar chart for discrete dimensions (supports grouped/stacked)
        - "line": Line chart, especially good for time series
        - "scatter": Scatter plot for correlation analysis
        - "heatmap": Heatmap for correlation matrices or pivot data
    
    Returns:
        Dictionary containing chart metadata and MCP image resource for Claude Desktop display
        
    Examples:
        # First get data with execute_sql_query, then create chart
        query_results = execute_sql_query("SELECT category, sales_amount FROM sales")
        create_chart(
            data_source=query_results["data"],
            chart_type="bar",
            x_column="category",
            y_column="sales_amount",
            title="Sales by Category"
        )
    """
    try:
        # Check for visualization libraries with detailed guidance
        missing_libs = []
        
        try:
            import pandas as pd
        except ImportError:
            missing_libs.append("pandas")
            
        if chart_library == "plotly":
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                from plotly.io import to_html, to_image
            except ImportError:
                missing_libs.append("plotly")
        else:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                plt.style.use('default')
            except ImportError as e:
                if "matplotlib" in str(e):
                    missing_libs.append("matplotlib")
                if "seaborn" in str(e):
                    missing_libs.append("seaborn")
        
        if missing_libs:
            return create_error_response(
                f"Missing required visualization libraries: {', '.join(missing_libs)}",
                "import_error",
                f"Install the missing libraries: pip install {' '.join(missing_libs)}. "
                f"If using a virtual environment, activate it first, then install the requirements: pip install -r requirements.txt"
            )
        
        # Validate input data
        if not data_source:
            return create_error_response(
                "No data provided for charting",
                "data_error"
            )
        
        data = data_source
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Validate required columns
        if x_column not in df.columns:
            return create_error_response(
                f"X-axis column '{x_column}' not found in data. Available columns: {list(df.columns)}",
                "column_error"
            )
        
        if chart_type in ["bar", "line", "scatter"] and y_column and y_column not in df.columns:
            return create_error_response(
                f"Y-axis column '{y_column}' not found in data. Available columns: {list(df.columns)}",
                "column_error"
            )
        
        # Generate title if not provided
        if not title:
            if chart_type == "bar":
                title = f"{y_column or 'Count'} by {x_column}"
            elif chart_type == "line":
                title = f"{y_column} over {x_column}"
            elif chart_type == "scatter":
                title = f"{y_column} vs {x_column}"
            elif chart_type == "heatmap":
                title = f"Heatmap of {x_column}" + (f" and {y_column}" if y_column else "")
            else:
                title = f"Chart of {x_column}"
        
        # Create chart ID for file naming and logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in title.replace(" ", "_") if c.isalnum() or c in "_-")
        chart_id = f"{chart_type}_{safe_title}_{timestamp}"
        
        # Generate chart and create PNG image for Claude Desktop
        image_data = None
        image_file_path = None
        
        if chart_library == "plotly":
            try:
                fig = _create_plotly_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width, height)
                # Try to export as PNG
                try:
                    image_bytes = fig.to_image(format='png', width=width, height=height, scale=2)
                    image_data = base64.b64encode(image_bytes).decode('utf-8')
                    # Save PNG to tmp directory
                    image_file_path = _save_image_to_tmp(image_bytes, chart_id, 'png')
                except Exception as e:
                    if "kaleido" in str(e).lower():
                        # Fallback to matplotlib if kaleido not available
                        chart_library = "matplotlib"
                    else:
                        raise e
            except ImportError:
                # Plotly not available, fall back to matplotlib
                chart_library = "matplotlib"
        
        if chart_library == "matplotlib":
            fig = _create_matplotlib_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width, height)
            # Generate PNG bytes with optimized settings
            import io
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                       facecolor='white', edgecolor='none', transparent=False,
                       dpi=150, pad_inches=0.1)
            image_bytes = img_buffer.getvalue()
            img_buffer.close()
            
            # Convert to base64 for Claude Desktop
            image_data = base64.b64encode(image_bytes).decode('utf-8')
            
            # Save PNG to tmp directory
            image_file_path = _save_image_to_tmp(image_bytes, chart_id, 'png')
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
        
        
        # Log chart creation success
        logger.info(f"Created {chart_type} chart with {len(df)} data points using {chart_library}")
        
        # Return the chart as PNG image for Claude Desktop
        if image_data:
            # Include file path info if image was saved
            file_info = f" (PNG saved to: {image_file_path})" if image_file_path else ""
            
            return [
                types.TextContent(
                    type="text", 
                    text=f"Generated {chart_type} chart with {len(df)} data points using {chart_library}.{file_info}"
                ),
                types.ImageContent(
                    type="image",
                    data=image_data,
                    mimeType="image/png"
                )
            ]
        else:
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Failed to generate chart image"
                )
            ]
        
    except Exception as e:
        logger.error(f"Chart creation error: {e}")
        # Return error as HTML resource
        error_html = f'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"></head>
<body style="margin:0;display:flex;align-items:center;justify-content:center;background:white;">
<div style="text-align:center;padding:20px;border:2px solid #dc2626;border-radius:8px;background:#fee2e2;color:#dc2626;max-width:400px;">
<h2>❌ Chart Generation Failed</h2>
<p>{str(e)[:100]}{"..." if len(str(e)) > 100 else ""}</p>
</div>
</body></html>'''
        
        return [
            types.TextContent(
                type="text",
                text=f"❌ Chart generation failed: {str(e)}"
            )
        ]

def _save_image_to_tmp(image_bytes: bytes, chart_id: str, format: str) -> str:
    """Save image bytes to temporary directory and return file path."""
    try:
        image_filename = f"{chart_id}.{format}"
        image_file_path = TMP_DIR / image_filename
        
        with open(image_file_path, 'wb') as f:
            f.write(image_bytes)
        
        logger.debug(f"Saved {format.upper()} chart to: {image_file_path}")
        return str(image_file_path)
    except Exception as e:
        logger.warning(f"Failed to save {format.upper()} file: {e}")
        return None

def _save_svg_to_tmp(svg_content: str, chart_id: str) -> str:
    """Save SVG content to temporary directory and return file path."""
    try:
        svg_filename = f"{chart_id}.svg"
        svg_file_path = TMP_DIR / svg_filename
        
        with open(svg_file_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        logger.debug(f"Saved SVG chart to: {svg_file_path}")
        return str(svg_file_path)
    except Exception as e:
        logger.warning(f"Failed to save SVG file: {e}")
        return None

def _create_react_chart_component(df, chart_type, x_column, y_column, color_column, title, width=800, height=600):
    """Create React component code for chart rendering."""
    import json
    
    # Convert DataFrame to JSON for React component
    data = df.to_dict('records')
    data_json = json.dumps(data, indent=2, default=str)
    
    # Generate appropriate chart component based on type
    if chart_type == "bar":
        return f"""import React from 'react';
import {{ BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer }} from 'recharts';

const ChartComponent = () => {{
  const data = {data_json};

  return (
    <div style={{{{ width: '100%', height: '{height}px', padding: '20px' }}}}>
      <h2 style={{{{ textAlign: 'center', marginBottom: '20px' }}}}>{title or f'{y_column or "Value"} by {x_column}'}</h2>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={{data}} margin={{{{ top: 20, right: 30, left: 20, bottom: 5 }}}}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="{x_column}" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="{y_column or 'value'}" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}};

export default ChartComponent;"""

    elif chart_type == "line":
        return f"""import React from 'react';
import {{ LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer }} from 'recharts';

const ChartComponent = () => {{
  const data = {data_json};

  return (
    <div style={{{{ width: '100%', height: '{height}px', padding: '20px' }}}}>
      <h2 style={{{{ textAlign: 'center', marginBottom: '20px' }}}}>{title or f'{y_column} over {x_column}'}</h2>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={{data}} margin={{{{ top: 20, right: 30, left: 20, bottom: 5 }}}}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="{x_column}" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="{y_column or 'value'}" stroke="#8884d8" strokeWidth={{2}} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}};

export default ChartComponent;"""

    elif chart_type == "scatter":
        return f"""import React from 'react';
import {{ ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer }} from 'recharts';

const ChartComponent = () => {{
  const data = {data_json};

  return (
    <div style={{{{ width: '100%', height: '{height}px', padding: '20px' }}}}>
      <h2 style={{{{ textAlign: 'center', marginBottom: '20px' }}}}>{title or f'{y_column} vs {x_column}'}</h2>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart data={{data}} margin={{{{ top: 20, right: 30, left: 20, bottom: 5 }}}}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="{x_column}" name="{x_column}" />
          <YAxis dataKey="{y_column or 'value'}" name="{y_column or 'value'}" />
          <Tooltip cursor={{{{ strokeDasharray: '3 3' }}}} />
          <Scatter fill="#8884d8" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}};

export default ChartComponent;"""

    else:  # Default to bar chart
        return f"""import React from 'react';
import {{ BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer }} from 'recharts';

const ChartComponent = () => {{
  const data = {data_json};

  return (
    <div style={{{{ width: '100%', height: '{height}px', padding: '20px' }}}}>
      <h2 style={{{{ textAlign: 'center', marginBottom: '20px' }}}}>{title or f'Chart of {x_column}'}</h2>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={{data}} margin={{{{ top: 20, right: 30, left: 20, bottom: 5 }}}}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="{x_column}" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="{y_column or 'value'}" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}};

export default ChartComponent;"""

def _create_plotly_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width=800, height=600):
    """Create Plotly chart based on type."""
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    
    if chart_type == "bar":
        if chart_style == "stacked":
            fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title, 
                        barmode='stack')
        else:  # grouped
            fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title, 
                        barmode='group')
    elif chart_type == "line":
        fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
        # Enhance for time series
        if df[x_column].dtype in ['datetime64[ns]', 'object']:
            try:
                df[x_column] = pd.to_datetime(df[x_column])
                fig.update_xaxes(title=x_column, type='date')
            except:
                pass
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title,
                        size_max=15)
    elif chart_type == "heatmap":
        if y_column:
            # Pivot table heatmap
            pivot_df = df.pivot_table(index=x_column, columns=y_column, aggfunc='size', fill_value=0)
        else:
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=['number']).columns
            pivot_df = df[numeric_cols].corr()
        
        fig = px.imshow(pivot_df, title=title, aspect="auto")
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    # Check if x-axis labels are long and need rotation
    if chart_type in ["bar", "line", "scatter"]:
        x_labels = df[x_column].astype(str).unique()
        max_label_length = max([len(str(label)) for label in x_labels]) if len(x_labels) > 0 else 0
        
        if max_label_length > 10 or len(x_labels) > 8:
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
    
    # Apply consistent styling
    fig.update_layout(
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(b=100, t=60, l=60, r=60),  # Add margins for labels
        showlegend=True if color_column else False,
        width=width,
        height=height
    )
    
    return fig

def _create_matplotlib_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width, height):
    """Create Matplotlib/Seaborn chart based on type."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set figure size
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    if chart_type == "bar":
        if color_column and chart_style == "stacked":
            pivot_df = df.pivot_table(index=x_column, columns=color_column, values=y_column, fill_value=0)
            pivot_df.plot(kind='bar', stacked=True, ax=ax)
        elif color_column:
            sns.barplot(data=df, x=x_column, y=y_column, hue=color_column, ax=ax)
        else:
            sns.barplot(data=df, x=x_column, y=y_column, ax=ax)
    elif chart_type == "line":
        if color_column:
            for group in df[color_column].unique():
                group_data = df[df[color_column] == group]
                ax.plot(group_data[x_column], group_data[y_column], label=group, marker='o')
            ax.legend()
        else:
            ax.plot(df[x_column], df[y_column], marker='o')
    elif chart_type == "scatter":
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=color_column, ax=ax, s=60)
    elif chart_type == "heatmap":
        if y_column:
            pivot_df = df.pivot_table(index=x_column, columns=y_column, aggfunc='size', fill_value=0)
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            pivot_df = df[numeric_cols].corr()
        sns.heatmap(pivot_df, annot=True, cmap='viridis', ax=ax)
    
    # Check if x-axis labels are long and need rotation
    if chart_type in ["bar", "line", "scatter"]:
        x_labels = [str(label) for label in ax.get_xticklabels()]
        if x_labels:  # Only if we have labels
            max_label_length = max([len(label.get_text()) for label in ax.get_xticklabels()]) if ax.get_xticklabels() else 0
            num_labels = len(ax.get_xticklabels())
            
            if max_label_length > 10 or num_labels > 8:
                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_column)
    if y_column:
        ax.set_ylabel(y_column)
    
    plt.tight_layout()
    return fig

# Image resource streaming no longer needed - SVG content returned directly

@mcp.tool()
def get_server_info() -> Dict[str, Any]:
    """Get information about the streamlined MCP server and its capabilities.
    
    Returns:
        Dictionary containing server information and available tools
    """
    return {
        "name": SERVER_NAME,
        "version": __version__, 
        "description": __description__,
        "supported_databases": ["PostgreSQL", "Snowflake"],
        "features": [
            "🌟 Single main analysis tool with automatic ontology generation",
            "🚀 5-step workflow (Connect → Analyze → Validate → Execute → Visualize)",
            "🎯 Self-sufficient ontologies with direct SQL references",
            "🔧 Ready-to-use JOIN conditions and business context",
            "📊 Interactive charting with Plotly and Matplotlib/Seaborn support",
            "⚡ Performance with consolidated functionality",
            "🎨 Clean, focused interface with 9 essential tools"
        ],
        "tools": [
            "connect_database",        # Connect to PostgreSQL or Snowflake database
            "list_schemas",           # List available database schemas  
            "get_analysis_context",   # 🚀 MAIN TOOL: Complete analysis with automatic ontology
            "generate_ontology",      # Generate ontology manually (if needed)
            "sample_table_data",      # Sample data from specific tables
            "validate_sql_syntax",    # Validate SQL before execution
            "execute_sql_query",      # Execute validated SQL queries
            "generate_chart",         # 📊 Generate interactive charts from data
            "get_server_info"         # Server information and capabilities
        ],
        "workflow": {
            "recommended_steps": [
                "1. connect_database() - Connect to your database",
                "2. get_analysis_context() - Get complete schema + ontology + relationships", 
                "3. validate_sql_syntax() - Validate your SQL queries",
                "4. execute_sql_query() - Execute validated queries",
                "5. generate_chart() - Visualize results with interactive charts"
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
            "log_level": server_config.log_level,
            "ontology_base_uri": server_config.ontology_base_uri,
            "max_query_limit": 5000,
            "supported_formats": ["turtle", "html", "png", "json"],
            "supported_chart_types": ["bar", "line", "scatter", "heatmap"],
            "supported_chart_libraries": ["plotly", "matplotlib"]
        }
    }
