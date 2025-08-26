#!/usr/bin/env python3
"""
Database Ontology MCP Server - Refactored

A focused MCP server with 11 essential tools for database analysis with automatic ontology generation and interactive charting.
Main tool: get_analysis_context() - provides complete schema analysis with integrated ontology.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastmcp import FastMCP
import mcp.types as types
from .config import config_manager
from . import __version__, __name__ as SERVER_NAME, __description__

# Import tool implementations from modules
from .tools import connection as conn_tools
from .tools import schema as schema_tools
from .tools import ontology as ontology_tools
from .tools import query as query_tools
from .tools import chart as chart_tools
from .tools import info as info_tools

# Initialize configuration and logging
server_config = config_manager.get_server_config()
logging.basicConfig(level=getattr(logging, server_config.log_level))
logger = logging.getLogger(__name__)

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

# =============================================================================
# MCP TOOLS - Decorated functions that delegate to tool modules
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
    role: Optional[str] = None,
    ssl: Optional[bool] = True
) -> Dict[str, Any]:
    """Connect to a PostgreSQL, Snowflake, or Dremio database.
    
    Parameters are optional - the tool will automatically use values from .env file when parameters are not provided.
    
    Args:
        db_type: Database type ("postgresql", "snowflake", or "dremio")
        host: Database host (PostgreSQL/Dremio, uses POSTGRES_HOST or DREMIO_HOST from .env if not provided)
        port: Database port (PostgreSQL/Dremio, uses POSTGRES_PORT or DREMIO_PORT from .env if not provided) 
        database: Database name (uses POSTGRES_DATABASE or SNOWFLAKE_DATABASE from .env if not provided)
        username: Username for authentication (uses POSTGRES_USERNAME, SNOWFLAKE_USERNAME, or DREMIO_USERNAME from .env if not provided)
        password: Password for authentication (uses POSTGRES_PASSWORD, SNOWFLAKE_PASSWORD, or DREMIO_PASSWORD from .env if not provided)
        account: Snowflake account identifier (Snowflake only, uses SNOWFLAKE_ACCOUNT from .env if not provided)
        warehouse: Snowflake warehouse (Snowflake only, uses SNOWFLAKE_WAREHOUSE from .env if not provided)
        schema: Schema name (Snowflake only, uses SNOWFLAKE_SCHEMA from .env if not provided, default: "PUBLIC")
        role: Snowflake role (Snowflake only, uses SNOWFLAKE_ROLE from .env if not provided, default: "PUBLIC")
        ssl: Enable SSL connection (Dremio only, default: True)
    
    Returns:
        Connection status information or error response
        
    Examples:
        # Connect using .env file values
        connect_database("postgresql")
        connect_database("snowflake")
        connect_database("dremio")
        
        # Override specific parameters
        connect_database("postgresql", host="custom.host.com", port=5433)
        connect_database("dremio", host="dremio.company.com", port=31010, ssl=False)
    """
    return conn_tools.connect_database(db_type, host, port, database, username, password, account, warehouse, schema, role, ssl)


@mcp.tool()
def diagnose_connection_issue(
    db_type: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    ssl: Optional[bool] = None
) -> Dict[str, Any]:
    """Diagnose connection issues and provide comprehensive troubleshooting guidance.
    
    This tool provides detailed connection diagnostics with specific recommendations
    for fixing common database connection problems. Includes connection testing steps,
    common issues, and detailed troubleshooting guidance.
    
    Args:
        db_type: Database type ("postgresql", "snowflake", or "dremio")
        host: Database host (optional, will use config if not provided)
        port: Database port (optional, will use config/default if not provided) 
        username: Username (optional, will use config if not provided)
        ssl: SSL setting for Dremio (optional, defaults to True)
    
    Returns:
        Comprehensive diagnostic information with troubleshooting recommendations
    """
    return conn_tools.diagnose_connection_issue(db_type, host, port, username, ssl)


@mcp.tool()
def list_schemas() -> Dict[str, Any]:
    """Get a list of available schemas from the connected database.
    
    Returns:
        List of schema names or error response
    """
    return schema_tools.list_schemas()


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
    return schema_tools.get_analysis_context(schema_name, include_ontology)


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
    return schema_tools.sample_table_data(table_name, schema_name, limit)


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
    return ontology_tools.generate_ontology(schema_name, base_uri, enrich_llm)


@mcp.tool()
def load_ontology_from_file(
    file_path: str
) -> Dict[str, Any]:
    """Load a previously saved or user-edited ontology from the tmp folder.
    
    This tool allows loading ontologies that were:
    - Previously generated and saved by get_analysis_context()  
    - Manually edited by users for enhanced analytical context
    - Created externally and placed in the tmp folder
    
    Args:
        file_path: Path to the ontology file (.ttl format)
                  Can be absolute path or relative to tmp folder
    
    Returns:
        Dictionary containing the loaded ontology content and metadata
        
    Examples:
        # Load a previously saved ontology
        load_ontology_from_file("ontology_public_20240826_143022.ttl")
        
        # Load with full path
        load_ontology_from_file("/path/to/tmp/ontology_custom.ttl")
    """
    return ontology_tools.load_ontology_from_file(file_path)


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
    return query_tools.validate_sql_syntax(sql_query)


@mcp.tool()
def execute_sql_query(
    sql_query: str, 
    limit: int = 1000
) -> Dict[str, Any]:
    """Execute a validated SQL query and return results safely.
    
    ## 🚀 **WORKFLOW** (Only 4 Steps):

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

    **3. 🛯f MANDATORY VALIDATION**
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

    ## 📝 COMMON DEADLY COMBINATIONS TO AVOID

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
    return query_tools.execute_sql_query(sql_query, limit)


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
    
    Chart Types:
        - "bar": Bar chart for discrete dimensions (supports grouped/stacked)
        - "line": Line chart, especially good for time series
        - "scatter": Scatter plot for correlation analysis
        - "heatmap": Heatmap for correlation matrices or pivot data
    
    Returns:
        MCP ContentBlock list with chart image for Claude Desktop display
        
    Examples:
        # First get data with execute_sql_query, then create chart
        query_results = execute_sql_query("SELECT category, sales_amount FROM sales")
        generate_chart(
            data_source=query_results["data"],
            chart_type="bar",
            x_column="category",
            y_column="sales_amount",
            title="Sales by Category"
        )
    """
    return chart_tools.generate_chart(data_source, chart_type, x_column, y_column, color_column, title, chart_library, chart_style, width, height)


@mcp.tool()
def get_server_info() -> Dict[str, Any]:
    """Get information about the Database Ontology MCP server and its capabilities.
    
    Returns:
        Dictionary containing server information and available tools
    """
    return info_tools.get_server_info()


if __name__ == "__main__":
    logger.info(f"Starting {SERVER_NAME} v{__version__}")
    logger.info(f"{__description__}")
    logger.info("=" * 60)
    logger.info("🔧 Available MCP Tools:")
    
    tools = [
        "connect_database - Connect to PostgreSQL, Snowflake, or Dremio with security",
        "diagnose_connection_issue - Diagnose and troubleshoot connection problems",
        "list_schemas - List available database schemas",
        "get_analysis_context - Complete schema analysis with automatic ontology generation", 
        "sample_table_data - Sample table data with security controls",
        "generate_ontology - Generate RDF ontology with validation",
        "load_ontology_from_file - Load saved/edited ontology from tmp folder",
        "validate_sql_syntax - Validate SQL queries before execution",
        "execute_sql_query - Execute validated SQL queries safely",
        "generate_chart - Generate interactive charts from query results",
        "get_server_info - Get comprehensive server information"
    ]
    
    for tool in tools:
        logger.info(f"  • {tool}")
    
    logger.info("")
    logger.info("🗄️ Supported Databases: PostgreSQL, Snowflake, Dremio")
    logger.info("🧠 LLM Enrichment: Available via MCP prompts and tools")
    logger.info("🔒 Security: Credential handling and input validation")
    logger.info("⚡ Performance: Connection pooling and parallel processing")
    logger.info("📊 Observability: Structured logging and comprehensive error handling")
    logger.info("")
    logger.info("📋 Configuration:")
    logger.info(f"  • Log Level: {server_config.log_level}")
    logger.info(f"  • Base URI: {server_config.ontology_base_uri}")
    logger.info("")
    logger.info("🚀 Starting MCP server with stdio transport...")
    logger.info("📡 Server ready for stdio MCP protocol messages")
    
    mcp.run()