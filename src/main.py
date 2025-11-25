"""Main MCP server application using FastMCP."""

import logging
import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

from dotenv import load_dotenv
from pydantic import BaseModel
from fastmcp import FastMCP, Context
from fastmcp.utilities.types import Image

from .database_manager import DatabaseManager, TableInfo, ColumnInfo
from .ontology_generator import OntologyGenerator
from .config import config_manager
from . import __version__, __name__ as SERVER_NAME

# Load environment variables from project root FIRST
# Try multiple possible paths for .env file
possible_env_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),  # relative to src
    os.path.join(os.getcwd(), '.env'),  # current working directory
    '/Users/ralfbecher/Documents/GitHub/mcp-servers/database-ontology-mcp/.env'  # absolute path
]

env_loaded = False
for env_path in possible_env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        env_loaded = True
        break

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log environment loading info
logger.info(f"Environment loading: tried {len(possible_env_paths)} paths, loaded: {env_loaded}")
if env_loaded:
    logger.info(f"POSTGRES_HOST from environment: {os.getenv('POSTGRES_HOST')}")
else:
    logger.warning("No .env file found - environment variables may not be available")

# --- MCP Server Setup ---

# Create server instance with comprehensive instructions
mcp = FastMCP(
    name=SERVER_NAME,
    instructions="""
# Orionbelt Semantic Layer - Database Ontology MCP Server

A sophisticated MCP server that provides **semantic database understanding** through ontology generation,
enabling accurate Text-to-SQL with automatic fan-trap prevention and relationship-aware query construction.

## 🎯 PURPOSE

This server transforms raw database schemas into **semantic ontologies** that provide:
- **Database schema linking** via RDF/OWL ontologies with db: namespace annotations
- **Relationship-aware SQL generation** with automatic JOIN condition inference
- **Fan-trap prevention** through relationship analysis and safe query patterns
- **Secure query execution** with syntax validation and injection prevention
- **Interactive data visualization** for analytical insights

## 🔧 CORE CAPABILITIES

### 1. Database Connectivity
- **PostgreSQL** - Full support with connection pooling
- **Snowflake** - Cloud data warehouse integration
- **Dremio** - Distributed query engine support (REST API)

### 2. Schema Intelligence
- Comprehensive table/column analysis with metadata
- Foreign key relationship mapping (critical for preventing data corruption)
- Primary key constraint identification
- Data type mapping and validation
- Row count statistics per table

### 3. Ontology Generation (Key Differentiator)
- **RDF/OWL ontology** creation from database schemas
- **db: namespace annotations** linking ontology classes to SQL tables/columns
- **Relationship preservation** capturing 1:1, 1:many, and many:many patterns
- **XSD type mapping** for proper data type handling
- Output in Turtle (.ttl) format with human-readable structure

### 4. Safe SQL Execution
- **Fan-trap detection** and prevention guidance
- **SQL injection protection** with pattern validation
- **Query validation** before execution
- **Controlled result limits** to prevent memory exhaustion
- **Execution monitoring** with performance metrics

### 5. Data Visualization
- Interactive and static chart generation (Matplotlib, Plotly)
- Bar charts, line plots, scatter plots, and heatmaps
- Direct integration with SQL query results
- Memory-efficient rendering without base64 encoding

## 📋 RECOMMENDED WORKFLOWS

### Workflow 1: Complete Schema Analysis → Ontology → SQL (RECOMMENDED)

**Purpose:** Generate accurate SQL with semantic context and fan-trap prevention

**Tool Chain:**
```
1. connect_database(db_type="postgresql")
   → Establish secure database connection

2. list_schemas()
   → Discover available schemas in the database

3. analyze_schema(schema_name="public")
   → Get complete schema structure with relationships
   → CRITICAL: Review foreign_keys for each table (fan-trap analysis)
   → Output includes: tables, columns, PKs, FKs, row counts

4. generate_ontology(schema_name="public")
   → Create RDF ontology with db: namespace linking
   → Ontology maps business concepts to SQL tables/columns
   → Preserves relationships for accurate JOIN generation
   → Saved to tmp/ directory for reference

5. execute_sql_query(sql_query="...", limit=1000)
   → Execute validated SQL with ontology context
   → Automatic fan-trap prevention guidance
   → Returns structured results with metadata
```

**Why This Order:**
- Schema analysis **must come before ontology** to capture relationships
- Ontology provides **semantic context** for accurate SQL generation
- Foreign key analysis **prevents fan-trap data corruption**
- Validation ensures **query safety** before execution

### Workflow 2: Quick Data Exploration

**Purpose:** Rapid data sampling and analysis without ontology

**Tool Chain:**
```
1. connect_database(db_type="snowflake")
2. list_schemas()
3. sample_table_data(table_name="customers", limit=10)
   → Quick data preview without full schema analysis
4. execute_sql_query(sql_query="SELECT COUNT(*) FROM public.customers")
```

### Workflow 3: SQL Validation → Execution → Visualization

**Purpose:** Validate, execute, and visualize analytical queries

**Tool Chain:**
```
1. validate_sql_syntax(sql_query="SELECT public.orders.category, SUM(public.orders.sales) FROM public.orders GROUP BY public.orders.category")
   → Syntax checking, security validation, performance analysis
   → Returns: is_valid, warnings, suggestions, security_analysis

2. execute_sql_query(sql_query="...", limit=500)
   → Execute if validation passes
   → Returns: data, columns, row_count, execution_time_ms

3. generate_chart(
     data_source=result['data'],
     chart_type='bar',
     x_column='category',
     y_column='sales'
   )
   → Create visualization from query results
   → Saved to tmp/ directory
```

### Workflow 4: Relationship Analysis for Complex Queries

**Purpose:** Prevent fan-traps when joining multiple fact tables

**Tool Chain:**
```
1. analyze_schema(schema_name="analytics")
   → EXAMINE foreign_keys field for EACH table
   → Identify 1:many relationships

2. ANALYZE for fan-traps:
   - Look for tables on "many" side of multiple relationships
   - Example: orders → order_items (1:many) + orders → shipments (1:many)
   - This is a FAN-TRAP: joining both will inflate aggregations

3. Use UNION ALL pattern (recommended):
   WITH unified_facts AS (
       SELECT ... FROM public.order_items
       UNION ALL
       SELECT ... FROM public.shipments
   )
   SELECT ... FROM unified_facts GROUP BY ...

4. execute_sql_query(sql_query="...")
   → Execute fan-trap-safe query
```

## ⚠️ CRITICAL: Fan-Trap Prevention

**What is a Fan-Trap?**
When a parent table has multiple 1:many relationships and you JOIN them with aggregation:
```
orders (1) → order_items (many)
orders (1) → shipments (many)

❌ WRONG: SELECT SUM(public.order_items.amount) FROM public.orders
          JOIN public.order_items ... JOIN public.shipments ...
          Result: Inflated totals due to Cartesian product

✅ RIGHT: Use UNION ALL to combine facts, then aggregate
```

**Always:**
1. Review `foreign_keys` from analyze_schema() FIRST
2. Use validate_sql_syntax() before execution
3. Use UNION ALL pattern for multi-fact queries
4. Validate results against source tables

## 🔐 SECURITY FEATURES

- **SQL injection prevention** - Pattern-based validation and parameterized queries
- **Query timeout protection** - Prevents runaway queries
- **Result size limits** - Configurable row limits (max 10,000)
- **Read-only enforcement** - No DML/DDL operations allowed
- **Credential encryption** - Secure password handling
- **Audit logging** - Security event tracking

## 🎓 ONTOLOGY-ENHANCED SQL GENERATION

**Key Advantage:** The generated ontology includes **db: namespace annotations** that map:
- Ontology classes → SQL table names
- Ontology properties → SQL column names
- Ontology relationships → SQL JOIN conditions

**Example Ontology Output:**
```turtle
:Customer a owl:Class ;
    rdfs:label "Customer" ;
    db:tableName "customers" ;          # Links to SQL table
    db:primaryKey "customer_id" .

:hasOrder a owl:ObjectProperty ;
    rdfs:domain :Customer ;
    rdfs:range :Order ;
    db:joinCondition "public.customers.customer_id = public.orders.customer_id" .  # JOIN hint
```

**This enables:**
- More accurate Text-to-SQL generation
- Automatic JOIN path discovery
- Relationship-aware query planning
- Prevention of incorrect table combinations

## 📊 SUPPORTED QUERY TYPES

- **SELECT** - Data retrieval with JOINs, aggregations, window functions
- **WITH (CTE)** - Common Table Expressions for complex queries
- **UNION/UNION ALL** - Combining result sets (recommended for fan-trap prevention)
- **Metadata queries** - DESCRIBE, SHOW, EXPLAIN
- **Analytical functions** - GROUP BY, HAVING, ORDER BY, LIMIT
- **Window functions** - OVER, PARTITION BY, ROW_NUMBER, RANK

## 💡 BEST PRACTICES

1. **Always start with schema analysis** - Understanding relationships prevents errors
2. **Generate ontology for semantic context** - Improves SQL accuracy significantly
3. **Validate before execution** - Catch errors early
4. **Use UNION ALL for multi-fact aggregation** - Prevents fan-trap inflation
5. **Check foreign_keys in schema output** - Critical for relationship understanding
6. **Apply sensible LIMIT values** - Start small, increase as needed
7. **Visualize results** - Charts reveal data patterns quickly

## 🚀 GETTING STARTED

**Minimal Example:**
```
1. connect_database(db_type="postgresql")
2. analyze_schema(schema_name="public")
3. generate_ontology(schema_name="public")
4. execute_sql_query(sql_query="SELECT * FROM public.customers LIMIT 10")
```

**Full Analytical Workflow:**
```
1. connect_database(db_type="snowflake")
2. list_schemas()
3. analyze_schema(schema_name="analytics")
4. generate_ontology(schema_name="analytics")
5. validate_sql_syntax(sql_query="...")
6. execute_sql_query(sql_query="...", limit=1000)
7. generate_chart(data_source=result['data'], chart_type='bar', ...)
```

## 📝 OUTPUT LOCATIONS

- **Ontologies**: Saved to `tmp/ontology_{{schema}}_{{timestamp}}.ttl`
- **Charts**: Saved to `tmp/chart_{{timestamp}}.png`
- **Logs**: Console output with INFO/WARNING/ERROR levels

## 🔗 TOOL CHAINING EXAMPLES

**Example 1 - Complete Analysis Pipeline:**
analyze_schema → generate_ontology → validate_sql_syntax → execute_sql_query → generate_chart

**Example 2 - Quick Exploration:**
connect_database → list_schemas → sample_table_data

**Example 3 - Query Optimization:**
analyze_schema → validate_sql_syntax (review warnings) → execute_sql_query

**Example 4 - Fan-Trap Safe Aggregation:**
analyze_schema (check FKs) → validate_sql_syntax (UNION pattern) → execute_sql_query

---

**Server Version**: {__version__}
**Supported Databases**: PostgreSQL, Snowflake, Dremio
**Primary Use Case**: Semantic database analysis with ontology-enhanced Text-to-SQL generation
""".format(__version__=__version__)
)


# --- Dependency Management ---

class ServerState:
    """Manages server state and dependencies."""

    def __init__(self):
        self._db_manager: Optional[DatabaseManager] = None
        self._ontology_generator: Optional[OntologyGenerator] = None
        self._loaded_ontology: Optional[str] = None  # Loaded ontology TTL content
        self._loaded_ontology_path: Optional[str] = None  # Path to loaded ontology file

    def get_db_manager(self) -> DatabaseManager:
        """Get or create database manager instance."""
        if self._db_manager is None:
            self._db_manager = DatabaseManager()
        return self._db_manager

    def get_ontology_generator(self, base_uri: str = "http://example.com/ontology/") -> OntologyGenerator:
        """Get ontology generator instance."""
        # Always create new instance to avoid state pollution
        return OntologyGenerator(base_uri=base_uri)

    def set_loaded_ontology(self, ontology_ttl: str, file_path: str):
        """Store a loaded ontology for use in context."""
        self._loaded_ontology = ontology_ttl
        self._loaded_ontology_path = file_path

    def get_loaded_ontology(self) -> Optional[str]:
        """Get the currently loaded ontology TTL content."""
        return self._loaded_ontology

    def get_loaded_ontology_path(self) -> Optional[str]:
        """Get the path to the currently loaded ontology file."""
        return self._loaded_ontology_path

    def has_loaded_ontology(self) -> bool:
        """Check if an ontology has been loaded."""
        return self._loaded_ontology is not None

    def cleanup(self):
        """Clean up resources."""
        if self._db_manager:
            self._db_manager.disconnect()
            self._db_manager = None
        self._ontology_generator = None
        self._loaded_ontology = None
        self._loaded_ontology_path = None

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
async def connect_database(ctx: Context, db_type: str) -> str:
    """Connect to a database using credentials from environment variables.
    
    Args:
        db_type: Database type - either 'postgresql', 'snowflake', or 'dremio'
    
    Returns:
        Connection status message or error JSON
    """
    # Validate input parameters
    if not db_type or db_type not in ["postgresql", "snowflake", "dremio"]:
        return create_error_response(
            f"Invalid database type '{db_type}'. Use 'postgresql', 'snowflake', or 'dremio'.",
            "validation_error"
        )
    
    db_manager = _server_state.get_db_manager()
    
    if db_type == "postgresql":
        # Get parameters from environment
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        database = os.getenv("POSTGRES_DATABASE")
        username = os.getenv("POSTGRES_USERNAME")
        password = os.getenv("POSTGRES_PASSWORD")
        
        # Validate required parameters
        required_params = {
            "POSTGRES_HOST": host,
            "POSTGRES_PORT": port,
            "POSTGRES_DATABASE": database,
            "POSTGRES_USERNAME": username,
            "POSTGRES_PASSWORD": password
        }
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            return create_error_response(
                f"Missing required environment variables for PostgreSQL: {', '.join(missing_params)}. Please check your .env file.",
                "validation_error"
            )
        
        success = db_manager.connect_postgresql(
            host=str(host),
            port=int(port),
            database=str(database),
            username=str(username),
            password=str(password)
        )
        db_name = database
        
    elif db_type == "snowflake":
        # Get parameters from environment
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        username = os.getenv("SNOWFLAKE_USERNAME")
        password = os.getenv("SNOWFLAKE_PASSWORD")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        database = os.getenv("SNOWFLAKE_DATABASE")
        schema = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        
        # Validate required parameters
        required_params = {
            "SNOWFLAKE_ACCOUNT": account,
            "SNOWFLAKE_USERNAME": username,
            "SNOWFLAKE_PASSWORD": password,
            "SNOWFLAKE_WAREHOUSE": warehouse,
            "SNOWFLAKE_DATABASE": database
        }
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            return create_error_response(
                f"Missing required environment variables for Snowflake: {', '.join(missing_params)}. Please check your .env file.",
                "validation_error"
            )
        
        success = db_manager.connect_snowflake(
            account=str(account),
            username=str(username),
            password=str(password),
            warehouse=str(warehouse),
            database=str(database),
            schema=schema
        )
        db_name = database
        
    elif db_type == "dremio":
        # Get parameters from environment
        host = os.getenv("DREMIO_HOST")
        port = os.getenv("DREMIO_PORT")
        username = os.getenv("DREMIO_USERNAME")
        password = os.getenv("DREMIO_PASSWORD")
        
        # Validate required parameters
        required_params = {
            "DREMIO_HOST": host,
            "DREMIO_PORT": port,
            "DREMIO_USERNAME": username,
            "DREMIO_PASSWORD": password
        }
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            return create_error_response(
                f"Missing required environment variables for Dremio: {', '.join(missing_params)}. Please check your .env file.",
                "validation_error"
            )
        
        # Dremio uses PostgreSQL protocol
        success = db_manager.connect_postgresql(
            host=str(host),
            port=int(port),
            database="DREMIO",  # Dremio typically uses this as default
            username=str(username),
            password=str(password)
        )
        db_name = "DREMIO"

    if success:
        await ctx.info(f"Database connected successfully; next call should be list_schemas or analyze_schema")
        return f"Successfully connected to {db_type} database: {db_name}"
    else:
        await ctx.info(f"Database connection failed; check credentials and try again")
        return create_error_response(
            f"Failed to connect to {db_type} database: {db_name}",
            "connection_error"
        )


@mcp.tool()
async def list_schemas(ctx: Context) -> List[str]:
    """Get a list of available schemas from the connected database.

    Returns:
        List of schema names or error response
    """
    db_manager = _server_state.get_db_manager()
    schemas = db_manager.get_schemas()
    if schemas:
        await ctx.info(f"Found {len(schemas)} schemas; next call should be analyze_schema")
    else:
        await ctx.info("No schemas found")
    return schemas if schemas else []


@mcp.tool()
async def analyze_schema(ctx: Context, schema_name: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a database schema and return comprehensive table information including relationships.

    This tool provides complete schema analysis including:
    - Table structure (columns, data types, nullability)
    - Primary key constraints
    - Foreign key relationships (critical for preventing fan-traps)
    - Table comments and metadata
    - Row counts for each table

    The analysis is automatically saved as a JSON file in the tmp/ folder for later use.

    RELATIONSHIP ANALYSIS:
    The foreign_keys field for each table contains all relationships, which is CRITICAL for:
    - Identifying 1:many relationships that can cause fan-traps
    - Understanding table dependencies for proper JOIN ordering
    - Detecting potential Cartesian product multiplications
    - Planning UNION ALL strategies for multiple fact tables

    IMPORTANT: This tool identifies schema relationships. When relationships include
    1:many patterns (e.g., sales → shipments), multi-fact queries MUST use UNION_ALL
    pattern to avoid data multiplication. See execute_sql_query() documentation for
    safe patterns.

    Args:
        schema_name: Name of the schema to analyze (optional)

    Returns:
        Dictionary containing:
        - schema: Schema name
        - table_count: Number of tables
        - tables: List of table dictionaries, each containing:
            - name: Table name
            - schema: Schema name
            - columns: List of column details (name, type, nullable, FK info)
            - primary_keys: List of primary key column names
            - foreign_keys: List of FK relationships (CRITICAL for fan-trap prevention)
                Each FK contains: column, referenced_table, referenced_column
            - comment: Table description
            - row_count: Number of rows in table
        - file_path: Path to the saved JSON file in tmp/ folder
        - next_steps: Recommended workflow guidance
        - analytical_guidance: Instructions for next step
    
    RECOMMENDED WORKFLOW:
    After analyzing schema, run generate_ontology() next for optimal SQL generation.
    
    Standard analytical workflow:
    1. analyze_schema() - Get schema structure and relationships
    2. generate_ontology() - Create ontology with database schema linking  
    3. execute_sql_query() - Generate SQL with ontology context
    
    The ontology provides:
    - Database schema linking (db: namespace annotations)
    - SQL column references (table.column format)
    - JOIN conditions for safe relationship traversal
    - Metadata for preventing fan-trap issues
    
    Ontology context improves SQL accuracy and helps prevent data corruption.
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
    
    schema_result = {
        "schema": schema_name or "default",
        "table_count": len(all_table_info),
        "tables": all_table_info
    }

    # Add analytical workflow guidance
    if all_table_info:
        schema_result["next_steps"] = {
            "recommended": "generate_ontology",
            "reason": "Generate ontology with database schema linking for accurate SQL generation and fan-trap prevention",
            "workflow": [
                "1. ✅ analyze_schema (completed)",
                "2. ➡️  generate_ontology (recommended next)",
                "3. ➡️  execute_sql_query (with ontology context)"
            ]
        }
        schema_result["analytical_guidance"] = (
            "Recommended next step: Run generate_ontology()\n\n"
            "This will create an ontology with:\n"
            "• Database schema linking (db: namespace)\n"
            "• SQL column references for queries\n"
            "• JOIN conditions for relationships\n"
            "• Metadata for fan-trap prevention\n\n"
            "The ontology provides context for accurate SQL generation."
        )
        schema_result["next_tool"] = "generate_ontology"
        await ctx.info(f"Schema analysis complete with {len(all_table_info)} tables; next call should be generate_ontology")
    else:
        await ctx.info("Schema analysis found no tables")

    # Save schema analysis to tmp folder for user access
    schema_file_path = None
    try:
        import json
        TMP_DIR = Path(__file__).parent.parent / "tmp"
        TMP_DIR.mkdir(exist_ok=True)  # Ensure tmp directory exists

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        schema_safe = (schema_name or "default").replace(" ", "_").replace(".", "_")
        schema_filename = f"schema_{schema_safe}_{timestamp}.json"
        schema_file_path = TMP_DIR / schema_filename

        with open(schema_file_path, 'w', encoding='utf-8') as f:
            json.dump(schema_result, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved schema analysis to: {schema_file_path}")

        # Add file path to result
        schema_result["file_path"] = str(schema_file_path)

    except Exception as e:
        logger.warning(f"Failed to save schema analysis to file: {e}")
        # Continue even if file save failed

    return schema_result


@mcp.tool()
async def generate_ontology(
    ctx: Context,
    schema_info: Optional[str] = None,
    schema_name: Optional[str] = None,
    base_uri: str = "http://example.com/ontology/"
) -> str:
    """Generate an RDF ontology from database schema information and stores it into a ttl file.
    
    Args:
        schema_info: JSON string containing schema information (tables, columns, relationships) 
                    If not provided, will attempt to fetch from connected database
        schema_name: Name of the schema to generate ontology from (optional)
        base_uri: Base URI for the ontology (default: http://example.com/ontology/)
    
    Returns:
        RDF ontology in Turtle format or error response
    """
    # Validate base_uri
    if not base_uri.endswith('/'):
        base_uri += '/'
    
    tables_info = []
    
    if schema_info:
        # Use provided schema information
        try:
            import json
            schema_data = json.loads(schema_info) if isinstance(schema_info, str) else schema_info

            # Convert schema data to TableInfo objects (already imported at module level)
            
            if "tables" in schema_data:
                for table_data in schema_data["tables"]:
                    # Convert column data
                    columns = []
                    for col_data in table_data.get("columns", []):
                        column = ColumnInfo(
                            name=col_data["name"],
                            data_type=col_data["data_type"],
                            is_nullable=col_data.get("is_nullable", True),
                            is_primary_key=col_data.get("is_primary_key", False),
                            is_foreign_key=col_data.get("is_foreign_key", False),
                            foreign_key_table=col_data.get("foreign_key_table"),
                            foreign_key_column=col_data.get("foreign_key_column"),
                            comment=col_data.get("comment")
                        )
                        columns.append(column)
                    
                    # Convert table data
                    table = TableInfo(
                        name=table_data["name"],
                        schema=table_data.get("schema", schema_name or "default"),
                        columns=columns,
                        primary_keys=table_data.get("primary_keys", []),
                        foreign_keys=table_data.get("foreign_keys", []),
                        comment=table_data.get("comment"),
                        row_count=table_data.get("row_count")
                    )
                    tables_info.append(table)
                    
            logger.info(f"Using provided schema info: {len(tables_info)} tables")
            
        except Exception as e:
            return create_error_response(
                f"Failed to parse schema_info parameter: {str(e)}",
                "parameter_error"
            )
    else:
        # Fall back to fetching from database
        db_manager = _server_state.get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established and no schema_info provided. Please use connect_database tool first or provide schema_info parameter.",
                "connection_error"
            )
        
        try:
            tables = db_manager.get_tables(schema_name)
            logger.info(f"Found {len(tables)} tables in schema '{schema_name or 'default'}': {tables}")
            
            for table_name in tables:
                try:
                    table_info = db_manager.analyze_table(table_name, schema_name)
                    if table_info:
                        tables_info.append(table_info)
                except Exception as e:
                    logger.error(f"Failed to analyze table {table_name}: {e}")
                    
        except Exception as e:
            return create_error_response(
                f"Failed to get tables from database: {str(e)}",
                "database_error"
            )

    if not tables_info:
        return create_error_response(
            f"No tables found to generate ontology from",
            "data_error"
        )

    generator = _server_state.get_ontology_generator(base_uri=base_uri)
    ontology_ttl = generator.generate_from_schema(tables_info)

    # Save ontology to tmp folder for user access
    ontology_file_path = None
    try:
        TMP_DIR = Path(__file__).parent.parent / "tmp"
        TMP_DIR.mkdir(exist_ok=True)  # Ensure tmp directory exists

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        schema_safe = (schema_name or "default").replace(" ", "_").replace(".", "_")
        ontology_filename = f"ontology_{schema_safe}_{timestamp}.ttl"
        ontology_file_path = TMP_DIR / ontology_filename
        
        with open(ontology_file_path, 'w', encoding='utf-8') as f:
            f.write(ontology_ttl)
        
        logger.info(f"Generated ontology for schema '{schema_name or 'default'}': {len(tables_info)} tables")
        logger.info(f"Saved ontology to: {ontology_file_path}")

        await ctx.info(f"Ontology generation complete; next call should be suggest_semantic_names to improve cryptic names")

        # Analyze the ontology for cryptic names
        generator = _server_state.get_ontology_generator(base_uri=base_uri)
        generator.graph.parse(data=ontology_ttl, format="turtle")
        # Re-bind namespaces after parsing
        generator.graph.bind("ns", generator.base_uri)
        generator.graph.bind("db", generator.db_ns)
        name_analysis = generator.extract_names_for_review()

        # Build result with guidance
        result = ontology_ttl
        result += f"\n\n# Ontology saved to: {ontology_file_path}"

        # Add semantic name resolution guidance if cryptic names detected
        cryptic_count = (name_analysis["summary"]["classes_needing_review"] +
                        name_analysis["summary"]["properties_needing_review"] +
                        name_analysis["summary"]["relationships_needing_review"])

        if cryptic_count > 0:
            result += f"\n\n# ⚠️ SEMANTIC NAME RESOLUTION RECOMMENDED"
            result += f"\n# Found {cryptic_count} names that may be abbreviations or cryptic identifiers."
            result += f"\n# To improve ontology readability for business users:"
            result += f"\n# 1. Call suggest_semantic_names() to extract names for review"
            result += f"\n# 2. Review the suggestions and provide business-friendly alternatives"
            result += f"\n# 3. Call apply_semantic_names() with your suggestions"
            result += f"\n#"
            result += f"\n# Analysis summary:"
            result += f"\n#   - Classes needing review: {name_analysis['summary']['classes_needing_review']}"
            result += f"\n#   - Properties needing review: {name_analysis['summary']['properties_needing_review']}"
            result += f"\n#   - Relationships needing review: {name_analysis['summary']['relationships_needing_review']}"

        return result

    except Exception as e:
        logger.warning(f"Failed to save ontology to file: {e}")
        await ctx.info(f"Ontology file save failed but ontology generated; next call should be suggest_semantic_names to improve cryptic names")
        # Still return the ontology even if file save failed
        return ontology_ttl


@mcp.tool()
async def suggest_semantic_names(
    ctx: Context,
    ontology_ttl: Optional[str] = None,
    schema_name: Optional[str] = None
) -> Dict[str, Any]:
    """Extract and analyze names from a generated ontology to identify abbreviations and cryptic names.

    This tool analyzes an ontology (either from a previous generate_ontology call or from the
    connected database) and returns a structured list of all class, property, and relationship
    names along with analysis of which ones appear to be abbreviations or cryptic identifiers.

    ## PURPOSE

    Since MCP Sampling is not available in Claude Desktop, this tool enables a two-step workflow:
    1. Call this tool to extract names that need improvement
    2. Review the results and provide suggested business-friendly names
    3. Call apply_semantic_names with your suggestions

    ## WORKFLOW

    After calling generate_ontology():
    ```
    1. suggest_semantic_names() → Returns list of names needing review
    2. LLM analyzes names and generates suggestions
    3. apply_semantic_names(suggestions) → Updates ontology with better names
    ```

    ## NAME ANALYSIS

    The tool automatically detects:
    - **Abbreviations**: Short names like 'cust', 'ord', 'amt', 'qty'
    - **Cryptic suffixes**: '_id', '_dt', '_cd', '_no', '_nm', '_flg'
    - **Technical prefixes**: 'pk_', 'fk_', 'tbl_', 'vw_'
    - **All-caps acronyms**: 'SKU', 'UPC', 'EAN'
    - **Numeric suffixes**: Names ending in numbers

    ## OUTPUT FORMAT

    Returns a dictionary with:
    - classes: List of table/class names with analysis
    - properties: List of column/property names with analysis
    - relationships: List of foreign key relationships with analysis
    - analysis_hints: Summary of detected issues
    - llm_prompt: Instructions for generating name suggestions

    Args:
        ontology_ttl: Optional Turtle format ontology string. If not provided,
                     will use the most recently generated ontology from the server state.
        schema_name: Optional schema name to regenerate ontology from database.
                    Only used if ontology_ttl is not provided.

    Returns:
        Dictionary containing extracted names, analysis results, and instructions
        for providing semantic name suggestions.

    Example Response Format for LLM to provide suggestions:
    ```json
    {
        "classes": [
            {"original_name": "cust_mstr", "suggested_name": "Customer", "description": "Master record for customer entities"}
        ],
        "properties": [
            {"original_name": "ord_dt", "table_name": "orders", "suggested_name": "Order Date", "description": "Date when order was placed"}
        ],
        "relationships": [
            {"original_name": "orders_has_customers", "suggested_name": "Placed By Customer", "description": "Links order to the customer who placed it"}
        ]
    }
    ```
    """
    try:
        generator = None

        if ontology_ttl:
            # Parse the provided ontology
            from rdflib import Graph
            generator = _server_state.get_ontology_generator()
            generator.graph = Graph()
            generator.graph.parse(data=ontology_ttl, format="turtle")
            # Re-bind namespaces after parsing
            generator.graph.bind("ns", generator.base_uri)
            generator.graph.bind("db", generator.db_ns)
        else:
            # Try to generate from database
            db_manager = _server_state.get_db_manager()

            if not db_manager.has_engine():
                return {
                    "error": "No ontology provided and no database connection. Please either provide ontology_ttl parameter or connect to database first.",
                    "error_type": "validation_error"
                }

            # Generate fresh ontology from database
            tables = db_manager.get_tables(schema_name)
            tables_info = []
            for table_name in tables:
                table_info = db_manager.analyze_table(table_name, schema_name)
                if table_info:
                    tables_info.append(table_info)

            if not tables_info:
                return {
                    "error": f"No tables found in schema '{schema_name or 'default'}'",
                    "error_type": "data_error"
                }

            generator = _server_state.get_ontology_generator()
            generator.generate_from_schema(tables_info)

        # Extract names for review
        extraction_result = generator.extract_names_for_review()

        # Add LLM prompt instructions
        extraction_result["llm_instructions"] = {
            "task": "Review the extracted names and provide business-friendly alternatives",
            "focus_on": [
                "Names marked with 'needs_review.is_cryptic: true'",
                "Abbreviations that should be expanded",
                "Technical names that need business context"
            ],
            "response_format": {
                "classes": [
                    {"original_name": "string", "suggested_name": "string", "description": "string"}
                ],
                "properties": [
                    {"original_name": "string", "table_name": "string", "suggested_name": "string", "description": "string"}
                ],
                "relationships": [
                    {"original_name": "string", "suggested_name": "string", "description": "string"}
                ]
            },
            "guidelines": [
                "Use clear, business-oriented terminology",
                "Expand abbreviations to full words (e.g., 'cust' → 'Customer')",
                "Use Title Case for class names",
                "Use descriptive phrases for properties",
                "Provide meaningful descriptions that explain business context",
                "Keep the original db:tableName and db:columnName for SQL generation"
            ]
        }

        # Add next step guidance
        extraction_result["next_step"] = "Review the names above and call apply_semantic_names with your suggestions"
        extraction_result["next_tool"] = "apply_semantic_names"

        await ctx.info(f"Extracted {extraction_result['summary']['total_classes']} classes, {extraction_result['summary']['total_properties']} properties for review; next call should be apply_semantic_names with your suggestions")

        return extraction_result

    except Exception as e:
        logger.error(f"Error extracting names for review: {e}")
        return {
            "error": f"Failed to extract names: {str(e)}",
            "error_type": "internal_error"
        }


@mcp.tool()
async def apply_semantic_names(
    ctx: Context,
    suggestions: str,
    schema_name: Optional[str] = None,
    save_to_file: bool = True
) -> str:
    """Apply LLM-suggested semantic names to the ontology.

    This tool takes name suggestions generated by the LLM (after calling suggest_semantic_names)
    and applies them to the ontology, updating labels and adding business descriptions.

    ## PURPOSE

    Completes the semantic name resolution workflow:
    1. generate_ontology() → Creates initial ontology with database names
    2. suggest_semantic_names() → Extracts names for LLM review
    3. apply_semantic_names() → Applies LLM suggestions to ontology

    ## WHAT GETS UPDATED

    For each suggestion, the tool will:
    - Update rdfs:label to the suggested business-friendly name
    - Add db:semanticName annotation with the new name
    - Add db:businessDescription with the provided description
    - Preserve original db:tableName/db:columnName for SQL generation

    ## IMPORTANT

    The original database identifiers (db:tableName, db:columnName) are preserved
    so that SQL generation continues to work correctly. The semantic names are
    used for business understanding and documentation.

    Args:
        suggestions: JSON string containing name suggestions in the format:
            {
                "classes": [{"original_name": "...", "suggested_name": "...", "description": "..."}],
                "properties": [{"original_name": "...", "table_name": "...", "suggested_name": "...", "description": "..."}],
                "relationships": [{"original_name": "...", "suggested_name": "...", "description": "..."}]
            }
        schema_name: Optional schema name to regenerate ontology from database before applying.
        save_to_file: Whether to save the updated ontology to a file (default: True)

    Returns:
        Updated ontology in Turtle format with semantic names applied.

    Example suggestions format:
    ```json
    {
        "classes": [
            {"original_name": "cust_mstr", "suggested_name": "Customer Master", "description": "Central repository of customer information"},
            {"original_name": "ord_hdr", "suggested_name": "Order Header", "description": "Main order record with summary information"}
        ],
        "properties": [
            {"original_name": "cust_id", "table_name": "cust_mstr", "suggested_name": "Customer ID", "description": "Unique identifier for customers"},
            {"original_name": "ord_dt", "table_name": "ord_hdr", "suggested_name": "Order Date", "description": "Date when the order was placed"}
        ],
        "relationships": [
            {"original_name": "ord_hdr_has_cust_mstr", "suggested_name": "Placed By", "description": "Links an order to the customer who placed it"}
        ]
    }
    ```
    """
    try:
        import json

        # Parse suggestions
        try:
            if isinstance(suggestions, str):
                name_suggestions = json.loads(suggestions)
            else:
                name_suggestions = suggestions
        except json.JSONDecodeError as e:
            return create_error_response(
                f"Invalid JSON in suggestions parameter: {str(e)}",
                "parameter_error",
                "Ensure suggestions is valid JSON with classes, properties, and relationships arrays"
            )

        # Validate structure
        if not isinstance(name_suggestions, dict):
            return create_error_response(
                "Suggestions must be a JSON object with 'classes', 'properties', and/or 'relationships' arrays",
                "parameter_error"
            )

        # Get or create ontology generator with existing graph
        db_manager = _server_state.get_db_manager()

        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established. Please use connect_database tool first.",
                "connection_error"
            )

        # Generate fresh ontology from database
        tables = db_manager.get_tables(schema_name)
        tables_info = []
        for table_name in tables:
            table_info = db_manager.analyze_table(table_name, schema_name)
            if table_info:
                tables_info.append(table_info)

        if not tables_info:
            return create_error_response(
                f"No tables found in schema '{schema_name or 'default'}'",
                "data_error"
            )

        generator = _server_state.get_ontology_generator()
        generator.generate_from_schema(tables_info)

        # Apply the semantic names
        updated_ontology = generator.apply_semantic_names(name_suggestions)

        # Save to file if requested
        ontology_file_path = None
        if save_to_file:
            try:
                TMP_DIR = Path(__file__).parent.parent / "tmp"
                TMP_DIR.mkdir(exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                schema_safe = (schema_name or "default").replace(" ", "_").replace(".", "_")
                ontology_filename = f"ontology_{schema_safe}_semantic_{timestamp}.ttl"
                ontology_file_path = TMP_DIR / ontology_filename

                with open(ontology_file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_ontology)

                logger.info(f"Saved semantic ontology to: {ontology_file_path}")

            except Exception as e:
                logger.warning(f"Failed to save ontology to file: {e}")

        # Count changes made
        classes_updated = len(name_suggestions.get("classes", []))
        properties_updated = len(name_suggestions.get("properties", []))
        relationships_updated = len(name_suggestions.get("relationships", []))
        total_updated = classes_updated + properties_updated + relationships_updated

        await ctx.info(f"Applied {total_updated} semantic name changes to ontology")

        result = f"# Semantic Names Applied Successfully\n\n"
        result += f"- Classes updated: {classes_updated}\n"
        result += f"- Properties updated: {properties_updated}\n"
        result += f"- Relationships updated: {relationships_updated}\n"

        if ontology_file_path:
            result += f"\n# Ontology saved to: {ontology_file_path}\n"

        result += f"\n{updated_ontology}"

        return result

    except Exception as e:
        logger.error(f"Error applying semantic names: {e}")
        return create_error_response(
            f"Failed to apply semantic names: {str(e)}",
            "internal_error"
        )


@mcp.tool()
async def load_my_ontology(
    ctx: Context,
    import_folder: str = "./import"
) -> Dict[str, Any]:
    """Load the newest .ttl ontology file from the import folder to use in context.

    This tool allows you to load a custom ontology file instead of generating one
    from the database schema. The loaded ontology will be used as the semantic
    context for subsequent operations like SQL generation and validation.

    ## PURPOSE

    Use this tool when you have a pre-existing ontology that should be used
    instead of auto-generating one from the database schema. This is useful for:
    - Using manually curated ontologies with business-friendly naming
    - Loading previously generated and refined ontologies
    - Working with ontologies that have been enhanced externally
    - Reusing ontologies across multiple sessions

    ## BEHAVIOR

    The tool will:
    1. Scan the specified import folder for .ttl (Turtle) files
    2. Select the newest file based on modification time
    3. Parse and validate the ontology
    4. Store it in server state for use in subsequent operations
    5. Return information about the loaded ontology

    ## FILE FORMAT

    The ontology file must be in Turtle (.ttl) format and should follow
    the RDF/OWL structure with:
    - owl:Class definitions for tables
    - owl:DatatypeProperty definitions for columns
    - owl:ObjectProperty definitions for relationships
    - db: namespace annotations for SQL mapping

    Args:
        import_folder: Path to the folder containing .ttl files (default: "./import")
                      Can be relative or absolute path.

    Returns:
        Dictionary containing:
        - success: Boolean indicating if ontology was loaded
        - file_path: Path to the loaded file
        - file_name: Name of the loaded file
        - file_size: Size of the file in bytes
        - modified_time: Last modification time of the file
        - classes_count: Number of OWL classes found
        - properties_count: Number of properties found
        - relationships_count: Number of object properties found
        - ontology_preview: First 2000 characters of the ontology
        - next_steps: Guidance for what to do next

    Example Usage:
        # Load ontology from default import folder
        load_my_ontology()

        # Load from custom folder
        load_my_ontology(import_folder="/path/to/my/ontologies")

    After Loading:
        The loaded ontology is stored in server state and will be available
        for reference during SQL generation and other semantic operations.
        You can view the full ontology content from the returned preview
        or by reading the file directly.
    """
    try:
        import glob
        from rdflib import Graph
        from rdflib.namespace import RDF, OWL

        # Resolve the import folder path
        if import_folder.startswith("./"):
            # Relative path - resolve from project root
            project_root = Path(__file__).parent.parent
            folder_path = project_root / import_folder[2:]
        elif not os.path.isabs(import_folder):
            # Relative path without ./
            project_root = Path(__file__).parent.parent
            folder_path = project_root / import_folder
        else:
            folder_path = Path(import_folder)

        # Check if folder exists
        if not folder_path.exists():
            return {
                "success": False,
                "error": f"Import folder not found: {folder_path}",
                "error_type": "folder_not_found",
                "suggestion": "Create the folder and add .ttl files, or specify a different path"
            }

        # Find all .ttl files in the folder
        ttl_files = list(folder_path.glob("*.ttl"))

        if not ttl_files:
            return {
                "success": False,
                "error": f"No .ttl files found in: {folder_path}",
                "error_type": "no_files_found",
                "suggestion": "Add .ttl ontology files to the import folder"
            }

        # Sort by modification time (newest first)
        ttl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        newest_file = ttl_files[0]

        # Read the file content
        with open(newest_file, 'r', encoding='utf-8') as f:
            ontology_content = f.read()

        # Parse and validate the ontology
        graph = Graph()
        try:
            graph.parse(data=ontology_content, format="turtle")
        except Exception as parse_error:
            return {
                "success": False,
                "error": f"Failed to parse ontology file: {str(parse_error)}",
                "error_type": "parse_error",
                "file_path": str(newest_file),
                "suggestion": "Ensure the file is valid Turtle format"
            }

        # Count ontology elements
        classes_count = len(list(graph.subjects(RDF.type, OWL.Class)))
        datatype_props = len(list(graph.subjects(RDF.type, OWL.DatatypeProperty)))
        object_props = len(list(graph.subjects(RDF.type, OWL.ObjectProperty)))

        # Get file stats
        file_stat = newest_file.stat()
        modified_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        # Store the loaded ontology in server state
        _server_state.set_loaded_ontology(ontology_content, str(newest_file))

        logger.info(f"Loaded ontology from: {newest_file}")
        logger.info(f"Ontology contains: {classes_count} classes, {datatype_props} data properties, {object_props} object properties")

        await ctx.info(f"Ontology loaded successfully with {classes_count} classes; ready for SQL generation")

        # Prepare preview (first 2000 chars)
        preview = ontology_content[:2000]
        if len(ontology_content) > 2000:
            preview += "\n\n... [truncated, full content available in file]"

        return {
            "success": True,
            "file_path": str(newest_file),
            "file_name": newest_file.name,
            "file_size": file_stat.st_size,
            "modified_time": modified_time,
            "classes_count": classes_count,
            "properties_count": datatype_props,
            "relationships_count": object_props,
            "total_files_found": len(ttl_files),
            "other_files": [f.name for f in ttl_files[1:5]] if len(ttl_files) > 1 else [],
            "ontology_preview": preview,
            "next_steps": {
                "recommended": "execute_sql_query",
                "reason": "The loaded ontology provides semantic context for SQL generation",
                "workflow": [
                    "1. ✅ load_my_ontology (completed)",
                    "2. ➡️  connect_database (if not already connected)",
                    "3. ➡️  execute_sql_query (use ontology context for accurate SQL)"
                ]
            },
            "note": "This ontology is now active and will be used instead of auto-generated ontologies"
        }

    except Exception as e:
        logger.error(f"Error loading ontology: {e}")
        return {
            "success": False,
            "error": f"Failed to load ontology: {str(e)}",
            "error_type": "internal_error"
        }


@mcp.tool()
async def sample_table_data(
    ctx: Context,
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

    if sample_data and len(sample_data) > 0:
        await ctx.info(f"Sample data retrieved with {len(sample_data)} rows; explore data or continue with other analysis")
    else:
        await ctx.info("No sample data found for table")

    return sample_data



@mcp.tool()
async def validate_sql_syntax(ctx: Context, sql_query: str) -> Dict[str, Any]:
    """Validate SQL query syntax without executing it, providing comprehensive analysis and suggestions.
    
    This tool performs thorough SQL syntax validation and analysis before query execution,
    helping prevent errors and optimize query performance across different database platforms.
    
    VALIDATION FEATURES:
    • Multi-database syntax checking (PostgreSQL, Snowflake, Dremio dialects)
    • Reserved keyword detection and escaping recommendations
    • Table and column existence verification against connected database schema
    • JOIN condition analysis and optimization suggestions
    • Subquery and CTE (Common Table Expression) validation
    • Aggregate function usage verification
    • Window function syntax checking
    
    SECURITY ANALYSIS:
    • SQL injection pattern detection
    • Potentially dangerous operation identification
    • Parameter binding recommendations
    • Access pattern analysis for sensitive tables
    
    PERFORMANCE OPTIMIZATION:
    • Index usage recommendations based on WHERE clauses
    • JOIN order optimization suggestions
    • Query complexity analysis and simplification recommendations
    • Estimated execution cost analysis
    • Memory usage predictions for large result sets
    
    COMPLIANCE CHECKING:
    • Read-only operation verification (no DML/DDL operations)
    • Column-level access permission validation
    • Data classification compliance (PII, sensitive data handling)
    • Query audit trail preparation
    
    Args:
        sql_query: The SQL query to validate. Can be any SELECT statement or
                  schema introspection query (SHOW, DESCRIBE, EXPLAIN).
    
    Returns:
        Dictionary containing:
        - is_valid: Boolean indicating if the query syntax is valid
        - database_dialect: Detected or assumed database dialect
        - validation_results: Detailed analysis of query components
        - suggestions: List of optimization and improvement recommendations
        - warnings: Potential issues or performance concerns
        - errors: Specific syntax errors with line/column information
        - estimated_complexity: Query complexity rating (low/medium/high)
        - security_analysis: Security-related findings and recommendations
        - table_references: List of tables and columns referenced in the query
        - required_permissions: Database permissions needed to execute the query
    
    Example Usage:
        # Validate a simple query
        validate_sql_syntax("SELECT * FROM public.customers WHERE public.customers.age > 25")

        # Validate a complex analytical query
        validate_sql_syntax(\"\"\"
            WITH monthly_sales AS (
                SELECT
                    DATE_TRUNC('month', public.orders.order_date) as month,
                    SUM(public.orders.amount) as total_sales
                FROM public.orders
                WHERE public.orders.order_date >= '2023-01-01'
                GROUP BY DATE_TRUNC('month', public.orders.order_date)
            )
            SELECT
                monthly_sales.month,
                monthly_sales.total_sales,
                LAG(monthly_sales.total_sales) OVER (ORDER BY monthly_sales.month) as prev_month_sales,
                ROUND((monthly_sales.total_sales - LAG(monthly_sales.total_sales) OVER (ORDER BY monthly_sales.month)) /
                      LAG(monthly_sales.total_sales) OVER (ORDER BY monthly_sales.month) * 100, 2) as growth_rate
            FROM monthly_sales
            ORDER BY monthly_sales.month
        \"\"\")
    
    Validation Categories:
        - SYNTAX: Basic SQL syntax compliance
        - SEMANTICS: Logical query structure and table/column references
        - SECURITY: Potential security vulnerabilities or risky patterns
        - PERFORMANCE: Query optimization opportunities
        - COMPLIANCE: Adherence to organizational data access policies
    """
    try:
        db_manager = _server_state.get_db_manager()
        
        if not db_manager.has_engine():
            return {
                "is_valid": False,
                "error": "No database connection established. Cannot perform full validation without schema information.",
                "error_type": "connection_error",
                "suggestions": [
                    "Use connect_database tool first to enable comprehensive validation",
                    "Basic syntax validation can still be performed, but schema validation requires a connection"
                ],
                "warnings": ["Schema-level validation disabled without database connection"],
                "database_dialect": "unknown"
            }
        
        # Validate SQL query is not empty
        if not sql_query or not sql_query.strip():
            return {
                "is_valid": False,
                "error": "SQL query cannot be empty.",
                "error_type": "parameter_error",
                "suggestions": ["Provide a valid SELECT statement or schema introspection query"],
                "database_dialect": "unknown"
            }
        
        # Perform validation through database manager
        validation_result = db_manager.validate_sql_syntax(sql_query.strip())

        # Log validation results
        if validation_result.get('is_valid'):
            logger.info(f"SQL validation successful: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")
            validation_result["next_tool"] = "execute_sql_query"
            await ctx.info("SQL validation passed; next call should be execute_sql_query")
        else:
            logger.info(f"SQL validation failed: {validation_result.get('error', 'Unknown validation error')}")
            await ctx.info("SQL validation failed; fix the query and try validate_sql_syntax again")

        return validation_result
        
    except Exception as e:
        logger.error(f"SQL validation error: {e}")
        return {
            "is_valid": False,
            "error": f"Validation system error: {str(e)}",
            "error_type": "internal_error",
            "suggestions": [
                "Check if the database connection is stable",
                "Verify the SQL query contains valid UTF-8 characters",
                "Try breaking down complex queries into smaller parts"
            ],
            "database_dialect": "unknown"
        }


@mcp.tool()
async def execute_sql_query(
    ctx: Context,
    sql_query: str,
    limit: int = 1000,
    checklist_completed: bool = False
) -> Dict[str, Any]:
    """Execute a validated SQL query against the connected database with comprehensive safety features.
    
    ## SQL TRAP PREVENTION

    ### REQUIRED: PRE-EXECUTION CHECKLIST

    You MUST complete these steps before calling this tool:

    1. ✓ RELATIONSHIP ANALYSIS - REQUIRED
        - Call analyze_schema() first
        - Examine the foreign_keys output
        - Identify all 1:many relationships
        - If you detect multiple 1:many relatione in the data model graph of the query, proceed to STEP 2

    2. ✓ FAN-TRAP DETECTION - REQUIRED
        - Pattern detected: multiple 1:many + aggregation?
        - If YES: You MUST use UNION_ALL (see PATTERN 1)
        - If NO: You may use other patterns

    3. ✓ SAFE PATTERN SELECTION - REQUIRED
        - Multiple fact tables? → Use UNION_ALL
        - Single fact table? → Use JOIN
        - Declare your pattern in the function call

    ## IDENTIFIER QUALIFICATION - CRITICAL

    **ALWAYS fully qualify table and column identifiers with schema prefix to avoid ambiguity and errors:**

    ✅ CORRECT:
    ```sql
    SELECT
        public.customers.customer_id,
        public.customers.name,
        public.orders.order_date
    FROM public.customers
    JOIN public.orders ON public.customers.customer_id = public.orders.customer_id
    ```

    ❌ WRONG:
    ```sql
    SELECT customer_id, name, order_date
    FROM customers
    JOIN orders ON customers.customer_id = orders.customer_id
    ```

    **Why this matters:**
    - Prevents ambiguous column references in JOINs
    - Avoids schema search path issues across different database systems
    - Ensures queries work consistently regardless of current schema context
    - Makes queries more maintainable and self-documenting
    - Required for cross-schema queries

    **Best practices:**
    - Use format: `schema_name.table_name.column_name` in SELECT clauses
    - Use format: `schema_name.table_name` in FROM and JOIN clauses
    - Qualify ALL identifiers, even when querying a single table
    - Use table aliases with qualified names for readability:
      ```sql
      SELECT c.customer_id, o.order_date
      FROM public.customers AS c
      JOIN public.orders AS o ON c.customer_id = o.customer_id
      ```

    ## SAFE QUERY PATTERNS

    ### PATTERN 1 - UNION APPROACH (RECOMMENDED FOR MULTI-FACT)

        - Always use UNION ALL for multi-fact table queries to prevent fan-traps
        - Prefer UNION approach over complex JOINs for dimensional analysis

    **Best for:** Multiple fact tables (sales, shipments, returns, etc.)

    ```sql
    WITH unified_facts AS (
        -- Sales facts
        SELECT
            public.sales.client_id,
            public.sales.product_id,
            public.sales.sales_amount as amount,
            0 as shipment_qty,
            0 as return_qty,
            'SALES' as fact_type
        FROM public.sales

        UNION ALL

        -- Shipment facts
        SELECT
            public.sales.client_id,
            public.sales.product_id,
            0 as amount,
            public.shipments.shipment_quantity,
            0 as return_qty,
            'SHIPMENT' as fact_type
        FROM public.shipments
        JOIN public.sales ON public.shipments.sales_id = public.sales.id

        UNION ALL

        -- Return facts
        SELECT
            public.sales.client_id,
            public.sales.product_id,
            0 as amount,
            0 as shipment_qty,
            public.returns.return_quantity,
            'RETURN' as fact_type
        FROM public.returns
        JOIN public.sales ON public.returns.sales_id = public.sales.id
    )
    SELECT
        unified_facts.client_id,
        unified_facts.product_id,
        SUM(unified_facts.amount) as total_sales,
        SUM(unified_facts.shipment_qty) as total_shipped,
        SUM(unified_facts.return_qty) as total_returned
    FROM unified_facts
    GROUP BY unified_facts.client_id, unified_facts.product_id;
    ```

    **Advantages:**
    - Natural fan-trap immunity by design
    - Unified data model for consistent aggregation
    - Easy to extend with additional fact types
    - Single aggregation logic for all measures
    - Better performance with fewer table scans

    ### PATTERN 2 - SEPARATE AGGREGATION (LEGACY APPROACH)

    **Use when:** UNION approach is not suitable or not needed (single fact)

    ```sql
    WITH fact1_totals AS (
        SELECT
            public.fact1.key,
            SUM(public.fact1.amount) as total_amount
        FROM public.fact1
        GROUP BY public.fact1.key
    ),
    fact2_totals AS (
        SELECT
            public.fact2.key,
            SUM(public.fact2.quantity) as total_quantity
        FROM public.fact2
        GROUP BY public.fact2.key
    )
    SELECT
        f1.key,
        f1.total_amount,
        COALESCE(f2.total_quantity, 0) as total_quantity
    FROM fact1_totals f1
    LEFT JOIN fact2_totals f2 ON f1.key = f2.key;
    ```

    ### PATTERN 3 - DISTINCT AGGREGATION (USE CAREFULLY)

    **Warning:** Only use when you fully understand the data relationships

    ```sql
    SELECT
        public.fact1.key,
        SUM(DISTINCT public.fact1.amount) as total_amount,
        SUM(public.fact2.quantity) as total_quantity
    FROM public.fact1
    LEFT JOIN public.fact2 ON public.fact1.id = public.fact2.fact1_id
    GROUP BY public.fact1.key;
    ```

    ### PATTERN 4 - WINDOW FUNCTIONS

    **For:** Complex analytical queries with preserved granularity

    ```sql
    SELECT DISTINCT
        public.fact1.key,
        SUM(public.fact1.amount) OVER (PARTITION BY public.fact1.key) as total_amount,
        f2.pre_aggregated_quantity
    FROM public.fact1
    LEFT JOIN (
        SELECT
            public.fact2.key,
            SUM(public.fact2.qty) as pre_aggregated_quantity
        FROM public.fact2
        GROUP BY public.fact2.key
    ) f2 ON public.fact1.key = f2.key;
    ```

    ## RESULT VALIDATION

    **Verify results make business sense:**
    - Compare totals with business expectations
    - Cross-check: `SELECT SUM(public.base_table.amount) FROM public.base_table` vs your query result
    - Ensure row counts are reasonable
    - High results may indicate fan-trap multiplication

    ## COMMON PROBLEMATIC COMBINATIONS

    **Patterns requiring careful review:**
    - `public.sales LEFT JOIN public.shipments + SUM(public.sales.amount)`
    - `public.orders LEFT JOIN public.order_items LEFT JOIN public.products + SUM(public.orders.total)`
    - `public.customers LEFT JOIN public.transactions LEFT JOIN public.transaction_items + aggregation`
    - Queries joining parent→child1 + parent→child2 with SUM/COUNT

    ## RELATIONSHIP EXAMPLES

    **Safe (1:1 relationships):**
    ```
    customers → customer_profiles (1:1)
    ```

    **Requires care (1:many):**
    ```
    customers → orders (1:many)
    ```

    **High risk (fan-trap potential):**
    ```
    orders → order_items (1:many) + orders → shipments (1:many)
    ```

    **For high-risk patterns:** Use UNION approach or separate aggregation CTEs

    ## FAN-TRAP SOLUTIONS

    If you suspect fan-trap in existing query:
    1. Split into UNION approach (recommended)
    2. Use separate aggregations  
    3. Add DISTINCT in SUM() as temporary fix
    4. Validate results against source tables
    5. Aggregate fact tables separately before joining

    **Note:** Fan-traps cause silent data corruption - queries execute successfully but return inflated results.

    ## VALIDATION CHECKLIST

    For queries with 2+ tables and aggregation:
    - Schema analyzed
    - Relationships reviewed
    - Fan-trap patterns checked
    - Syntax validated
    - Safe aggregation pattern used
    - Results validated
    
    This tool provides enterprise-grade SQL execution with multiple layers of protection:
    
    SECURITY FEATURES:
    • SQL injection prevention through parameterized queries and input sanitization
    • Query timeout protection to prevent runaway queries
    • Result set size limits to prevent memory exhaustion
    • Connection state validation before execution
    • Comprehensive error handling and logging
    
    QUERY VALIDATION:
    • Automatic syntax validation before execution
    • Database-specific dialect checking (PostgreSQL, Snowflake, Dremio)
    • Reserved keyword detection and escaping
    • Schema and table existence verification
    
    PERFORMANCE OPTIMIZATION:
    • Configurable result limits to control memory usage
    • Query execution time tracking
    • Connection pooling for optimal resource utilization
    • Structured result formatting for efficient data transfer
    
    OBSERVABILITY:
    • Detailed execution logging with query fingerprints
    • Performance metrics (execution time, rows affected, data transferred)
    • Error classification with diagnostic information
    • Query history tracking for audit purposes
    
    SUPPORTED QUERY TYPES:
    • SELECT statements (data retrieval with joins, aggregations, window functions)
    • Data exploration queries (DESCRIBE, SHOW, EXPLAIN)
    • Schema introspection (information_schema queries)
    • CTE (Common Table Expression) queries
    • Complex analytical queries with multiple joins
    
    RESULT FORMATTING:
    • JSON-structured response with metadata
    • Column type information and nullability
    • Row count and execution statistics
    • Error details with suggested fixes
    • Query fingerprint for caching and optimization
    
    ENTERPRISE SAFETY:
    • Read-only execution mode (no DML/DDL operations)
    • Resource consumption monitoring
    • Graceful degradation on connection issues
    • Detailed audit trail for compliance
    
    Args:
        sql_query: The SQL query to execute. Must be a valid SELECT or introspection query.
                  DML operations (INSERT, UPDATE, DELETE) and DDL operations (CREATE, DROP, ALTER)
                  are not permitted for security reasons.
        limit: Maximum number of rows to return (default: 1000, max: 10000).
              This prevents memory exhaustion from large result sets while allowing
              comprehensive data analysis.
        checklist_completed: Boolean flag (true/false, NOT "True"/"False" strings) confirming
                           that the pre-execution checklist has been completed (default: false).
                           This includes relationship analysis, fan-trap detection, and safe
                           pattern selection. Must be set to true to execute queries.
                           Type: boolean (not string)

    Returns:
        Dictionary containing:
        - success: Boolean indicating if query executed successfully
        - data: List of dictionaries representing query results (if successful)
        - columns: List of column metadata with names, types, and constraints
        - row_count: Number of rows returned
        - execution_time_ms: Query execution time in milliseconds
        - query_fingerprint: Unique identifier for the executed query
        - metadata: Additional query and connection information
        - error: Detailed error message (if unsuccessful)
        - error_type: Classification of error for programmatic handling
        - suggestions: Helpful suggestions for fixing query issues
    
    Example Usage:
        # Simple data retrieval
        execute_sql_query("SELECT * FROM public.customers WHERE public.customers.country = 'USA'", 100)

        # Complex analytical query
        execute_sql_query(\"\"\"
            SELECT
                public.products.category,
                COUNT(*) as product_count,
                AVG(public.products.price) as avg_price,
                SUM(public.sales.revenue) as total_revenue
            FROM public.products p
            JOIN public.sales s ON p.id = s.product_id
            WHERE s.date >= '2023-01-01'
            GROUP BY public.products.category
            ORDER BY total_revenue DESC
        \"\"\", 500)

        # Schema introspection
        execute_sql_query("SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public'")
    
    Error Handling:
        Returns structured error information for:
        - Syntax errors with specific line/column information
        - Permission errors with detailed explanations
        - Connection timeouts with retry suggestions
        - Resource exhaustion with optimization recommendations
        - Data type conflicts with conversion suggestions
    
    Performance Notes:
        - Queries are automatically optimized based on database dialect
        - Result sets are streamed for memory efficiency
        - Connection pooling reduces overhead for multiple queries
        - Query plans are cached for repeated similar queries
    """
    try:
        # Handle string "True"/"False" from LLMs that send strings instead of booleans
        if isinstance(checklist_completed, str):
            checklist_completed = checklist_completed.lower() in ('true', '1', 'yes')

        db_manager = _server_state.get_db_manager()

        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established. Please use connect_database tool first to establish a connection to PostgreSQL, Snowflake, or Dremio.",
                "connection_error",
                "Available connection methods: connect_database('postgresql'), connect_database('snowflake'), connect_database('dremio')"
            )

        # Validate limit parameter
        if limit <= 0 or limit > 10000:
            return create_error_response(
                f"Invalid limit value '{limit}'. Must be between 1 and 10000.",
                "parameter_error",
                "Use a reasonable limit to prevent memory exhaustion while allowing comprehensive analysis."
            )
        
        # Validate SQL query is not empty
        if not sql_query or not sql_query.strip():
            return create_error_response(
                "SQL query cannot be empty.",
                "parameter_error",
                "Provide a valid SELECT statement or schema introspection query."
            )

        # Validate checklist_completed parameter
        if not checklist_completed:
            return create_error_response(
                "ERROR: PRE-EXECUTION CHECKLIST NOT COMPLETED.\nSee tool description for required steps.",
                "validation_error",
                "You must complete the pre-execution checklist before executing SQL queries. Review the tool documentation for required steps."
            )

        # Execute the query through the database manager
        result = db_manager.execute_sql_query(sql_query.strip(), limit)

        # Log execution results
        if result.get('success'):
            logger.info(f"SQL query executed successfully: {result.get('row_count', 0)} rows returned in {result.get('execution_time_ms', 0)}ms")
            row_count = result.get('row_count', 0)
            if row_count > 0:
                result["next_tool"] = "generate_chart"
                await ctx.info(f"SQL query executed successfully with {row_count} rows; next call should be generate_chart for visualization")
            else:
                await ctx.info("SQL query executed successfully but returned no rows")
        else:
            logger.warning(f"SQL query execution failed: {result.get('error', 'Unknown error')}")
            await ctx.info("SQL query execution failed; review error and try again")

        return result
        
    except Exception as e:
        logger.error(f"Critical error in SQL execution: {e}")
        return create_error_response(
            f"Internal server error during SQL execution: {str(e)}",
            "internal_error",
            "This may indicate a system-level issue. Please check server logs and try again."
        )


@mcp.tool()
async def generate_chart(
    ctx: Context,
    data_source: Union[List[Dict[str, Any]], str],
    chart_type: str,
    x_column: str,
    y_column: Optional[Union[str, List[str]]] = None,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    chart_library: str = "matplotlib",
    chart_style: str = "grouped",
    width: int = 800,
    height: int = 600,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None
) -> Image:
    """Generate interactive charts from SQL query results or data analysis.

    ⚠️ IMPORTANT: data_source parameter MUST be valid JSON (array of objects with double quotes)
    NOT a string representation with single quotes. See Args section for correct format.

    📊 VISUALIZATION CAPABILITIES:
    This tool creates professional data visualizations directly in Claude Desktop,
    supporting both static (Matplotlib) and interactive (Plotly) chart libraries.

    CHART TYPES:
    • **bar**: Bar charts for categorical comparisons
      - Grouped bars for multi-series data (use color_column with chart_style='grouped')
      - Stacked bars for part-to-whole relationships (use color_column with chart_style='stacked')
      - ⚠️ y_column must contain numeric values
      - Default sorting: by measure (y_column) descending
    • **line**: Line charts for trends over time
      - Single measure with optional color grouping
      - Multiple measures for comparison (pass y_column as list of column names)
      - Automatic time series detection
      - ⚠️ y_column(s) must contain numeric values
      - Default sorting: by dimension (x_column) ascending
    • **scatter**: Scatter plots for correlations
      - Color coding by category
      - ⚠️ Both x_column and y_column must contain numeric values
    • **heatmap**: Heat maps for matrix data
      - Correlation matrices
      - Pivot table visualizations
      - Default sorting: x_column ascending, y_column descending

    LIBRARIES:
    • **matplotlib** (default): Static PNG charts
      - Better for simple visualizations
      - Guaranteed compatibility
      - Seaborn styling for aesthetics
    • **plotly**: Interactive HTML/PNG charts
      - Hover tooltips and zoom
      - Better for complex data exploration
      - Falls back to matplotlib if not available

    DATA PREPARATION:
    The tool automatically handles:
    - SQL query results from execute_sql_query
    - JSON data structures
    - Pandas DataFrame conversion
    - Missing value handling
    - Automatic type detection

    Args:
        data_source: **MUST BE VALID JSON** - Array of objects containing the data to visualize.
                    ⚠️ CRITICAL: Send as actual JSON array with double quotes, NOT a string representation.

                    ✅ CORRECT FORMAT (JSON):
                    [
                      {"country": "USA", "customer_count": 5},
                      {"country": "Germany", "customer_count": 3},
                      {"country": "UK", "customer_count": 4}
                    ]

                    ❌ WRONG FORMAT (String with single quotes):
                    "[{'country': 'USA', 'customer_count': 5}, ...]"

                    Typically the 'data' field from execute_sql_query results.
        chart_type: Type of chart - 'bar', 'line', 'scatter', or 'heatmap'
        x_column: Column name for X-axis (required)
        y_column: Column name(s) for Y-axis. Can be:
                  - String: single measure (all chart types)
                  - List of strings: multiple measures (line charts only - creates multi-line comparison)
                  ⚠️ IMPORTANT: Measure columns must contain numeric values (integers or floats)
        color_column: Column for color grouping/legend (optional)
                     - For bar charts: creates grouped or stacked bars based on chart_style
                     - For line/scatter: creates separate series with different colors
        title: Chart title (auto-generated if not provided)
        chart_library: 'matplotlib' or 'plotly' (default: matplotlib)
        chart_style: 'grouped' or 'stacked' for bar charts (default: grouped)
                    - 'grouped': bars side-by-side for comparison
                    - 'stacked': bars stacked on top of each other (requires color_column)
        width: Chart width in pixels (default: 800)
        height: Chart height in pixels (default: 600)
        sort_by: Column to sort by (optional). If not specified, uses automatic sorting:
                 - Bar/grouped/stacked: sorts by measure (y_column) descending
                 - Line: sorts by dimension (x_column) ascending
                 - Heatmap: sorts x_column ascending, y_column descending
        sort_order: 'ascending' or 'descending' (optional). If not specified, uses automatic order based on chart type.

    Returns:
        Image object that can be displayed in Claude Desktop

    Example Usage:
        # 1. Simple bar chart from query results
        result = execute_sql_query("SELECT public.orders.category, SUM(public.orders.sales) as total FROM public.orders GROUP BY public.orders.category")
        generate_chart(result['data'], 'bar', 'category', 'total')

        # Example with explicit JSON format (what data_source should look like):
        generate_chart(
            data_source=[
                {"country": "USA", "customer_count": 5},
                {"country": "Germany", "customer_count": 3},
                {"country": "UK", "customer_count": 4},
                {"country": "France", "customer_count": 2},
                {"country": "Japan", "customer_count": 1}
            ],
            chart_type="bar",
            x_column="country",
            y_column="customer_count"
        )

        # 2. Stacked bar chart with two dimensions
        result = execute_sql_query(\"\"\"
            SELECT
                public.sales.region,
                public.sales.product_type,
                SUM(public.sales.revenue) as total
            FROM public.sales
            GROUP BY public.sales.region, public.sales.product_type
        \"\"\")
        generate_chart(result['data'], 'bar', 'region', 'total', 'product_type', chart_style='stacked')

        # 3. Time series line chart with single measure
        result = execute_sql_query("SELECT public.daily_sales.date, public.daily_sales.revenue FROM public.daily_sales ORDER BY public.daily_sales.date")
        generate_chart(result['data'], 'line', 'date', 'revenue', title='Revenue Trend')

        # 4. Multi-measure line chart for comparison (NEW!)
        result = execute_sql_query("SELECT public.monthly_data.month, public.monthly_data.revenue, public.monthly_data.expenses, public.monthly_data.profit FROM public.monthly_data ORDER BY public.monthly_data.month")
        generate_chart(result['data'], 'line', 'month', ['revenue', 'expenses', 'profit'],
                      title='Financial Metrics Comparison')

        # 5. Grouped bar chart
        result = execute_sql_query(\"\"\"
            SELECT
                public.sales.region,
                public.sales.product,
                SUM(public.sales.quantity) as units
            FROM public.sales
            GROUP BY public.sales.region, public.sales.product
        \"\"\")
        generate_chart(result['data'], 'bar', 'region', 'units', 'product', chart_style='grouped')

        # 6. Correlation heatmap
        result = execute_sql_query("SELECT * FROM public.metrics")
        generate_chart(result['data'], 'heatmap', x_column='metric1')

        # 7. Scatter plot with categories
        result = execute_sql_query("SELECT public.products.price, public.products.quality, public.products.brand FROM public.products")
        generate_chart(result['data'], 'scatter', 'price', 'quality', 'brand')

    STYLING NOTES:
    - Long labels are automatically rotated for readability
    - Colors are chosen from professional palettes
    - Legends appear when color_column is specified
    - Grid lines and axes are optimized for clarity

    PERFORMANCE:
    - Charts are rendered as PNG and saved to tmp/ directory
    - Images are returned directly for display in Claude Desktop
    - Memory is properly managed (figures closed after rendering)
    - Large datasets are automatically sampled if needed

    ERROR HANDLING:
    - Missing libraries trigger helpful installation instructions
    - Invalid column names show available columns
    - Data type mismatches are automatically corrected
    - Fallback from Plotly to Matplotlib if dependencies missing
    """
    # Import the implementation from tools module
    from .tools.chart import generate_chart as generate_chart_impl
    import json

    # FALLBACK: Parse data_source if it's incorrectly sent as a string
    # (documentation specifies it should be actual JSON, but some LLMs might send strings)
    if data_source and isinstance(data_source, str):
        logger.warning("data_source was sent as string instead of JSON array - attempting to parse")
        try:
            # First try JSON parsing (double quotes)
            parsed = json.loads(data_source)
            if isinstance(parsed, list):
                data_source = parsed
                logger.info("Successfully parsed data_source from JSON string format")
        except (json.JSONDecodeError, ValueError):
            # If JSON fails, try Python literal_eval (single quotes)
            try:
                import ast
                parsed = ast.literal_eval(data_source)
                if isinstance(parsed, list):
                    data_source = parsed
                    logger.info("Successfully parsed data_source from Python literal string format")
            except (ValueError, SyntaxError) as e:
                # Could not parse - return clear error
                await ctx.info("Chart generation failed - data_source format error")
                raise RuntimeError(
                    f"❌ data_source must be valid JSON (array of objects), not a string. "
                    f"Received string: {data_source[:100]}... "
                    f"Expected format: [{{'key': 'value'}}, ...] "
                    f"Parse error: {str(e)}"
                )

    # Parse y_column if it's a JSON string representation of a list
    # This handles cases where MCP passes ["col1", "col2"] as a string
    if y_column and isinstance(y_column, str):
        try:
            parsed = json.loads(y_column)
            if isinstance(parsed, list):
                y_column = parsed
        except (json.JSONDecodeError, ValueError):
            # Not a JSON string, keep as-is (it's a column name)
            pass

    # Call the implementation to get image bytes and chart_id
    result = generate_chart_impl(
        data_source, chart_type, x_column, y_column, color_column,
        title, chart_library, chart_style, width, height, sort_by, sort_order
    )

    # Check if chart generation was successful
    if isinstance(result, dict) and result.get("error"):
        await ctx.info("Chart generation failed")
        raise RuntimeError(result.get("error", "Chart generation failed"))

    if isinstance(result, tuple) and len(result) == 2:
        image_bytes, chart_id = result

        # Save the image to tmp directory
        from .chart_utils import save_image_to_tmp
        image_file_path = save_image_to_tmp(image_bytes, chart_id, 'png')

        if not image_file_path:
            await ctx.info("Chart generation failed to save file")
            raise RuntimeError("Failed to save chart image to file")

        await ctx.info(f"Chart generated successfully: {image_file_path}")

        # Return Image object for Claude Desktop display
        return Image(path=str(image_file_path))
    else:
        await ctx.info("Chart generation failed")
        raise RuntimeError("Chart generation failed: unexpected result format")


@mcp.tool()
async def get_server_info(ctx: Context) -> Dict[str, Any]:
    """Get information about the MCP server and its capabilities.

    Returns:
        Dictionary containing server information
    """
    # Import centralized version info
    from . import __version__, __name__ as SERVER_NAME, __description__

    await ctx.info("Server info retrieved; next call should be connect_database to start working")

    return {
        "name": SERVER_NAME,
        "version": __version__,
        "description": __description__,
        "supported_databases": ["postgresql", "snowflake", "dremio"],
        "features": [
            "Database connection management",
            "Schema analysis",
            "Table relationship mapping",
            "RDF/OWL ontology generation",
            "Load custom ontologies from file",
            "Semantic name resolution (abbreviation expansion)",
            "Interactive data visualization (charts)"
        ],
        "tools": [
            "connect_database",
            "list_schemas",
            "analyze_schema",
            "generate_ontology",
            "suggest_semantic_names",
            "apply_semantic_names",
            "load_my_ontology",
            "sample_table_data",
            "validate_sql_syntax",
            "execute_sql_query",
            "generate_chart",
            "get_server_info"
        ],
        "next_tool": "connect_database"
    }


# --- Cleanup on shutdown ---

def cleanup_server():
    """Clean up server resources."""
    _server_state.cleanup()


# Main execution removed - server should only be started via server.py
# This prevents double startup when main.py is imported
