# Database Ontology MCP Server

🚀 **Streamlined MCP server for database analysis with automatic ontology generation and interactive charting - now with 9 essential tools for maximum effectiveness.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.10+-blue)](https://github.com/jlowin/fastmcp)

This project provides a focused, production-ready Python-based MCP (Model Context Protocol) server that analyzes relational database schemas (PostgreSQL and Snowflake) and automatically generates comprehensive ontologies in RDF/Turtle format with direct SQL mappings.

## 🎯 Key Philosophy: Automatic Ontology Integration

**Problem Solved**: Claude Desktop often ignores ontology generation tools during analytical sessions, treating them as optional rather than essential.

**Solution**: Our main analysis tool `get_analysis_context()` automatically includes ontology generation, making semantic context readily available for every query.

## 🌟 Key Features

### 🔗 Database Connectivity
- **PostgreSQL** and **Snowflake** support with connection pooling
- **Environment variable fallback** - parameters optional, uses .env when not provided
- **Enhanced connection management** with retry logic and timeout handling
- **Automatic dependency management** for Snowflake connectors

### 🎯 Streamlined Architecture (9 Essential Tools)
- **One main tool** (`get_analysis_context`) with automatic ontology generation
- **Interactive charting tool** (`create_chart`) for data visualization
- **Consolidated workflow** - no more tool confusion or bloat
- **Inline functionality** - reduces dependencies between tools
- **Focus on results** - maximum effectiveness with minimum complexity

### 🧠 Automatic Ontology Generation
- **Self-sufficient ontologies** with direct database references (`db:sqlReference`, `db:sqlJoinCondition`)
- **Business context inference** from table and column naming patterns
- **Complete SQL mappings** embedded directly in ontology
- **Fan-trap detection** and query safety validation

### 🛡️ Advanced SQL Safety
- **Fan-trap prevention protocols** with mandatory relationship analysis
- **Query pattern validation** to prevent data multiplication errors
- **Safe aggregation patterns** (UNION, separate CTEs, window functions)
- **Comprehensive SQL validation** before execution

### ⚡ Performance & Reliability
- **Concurrent processing** with thread pool management
- **Connection pooling** and resource optimization
- **Comprehensive error handling** with structured responses
- **Production-ready logging** and monitoring

## 📦 Python Library Installation

### Required Dependencies

```bash
# Install all required dependencies
pip install -r requirements.txt
```

### Complete Library List

The project uses the following Python libraries:

#### **Core MCP Framework**
```bash
fastmcp>=2.10.0,<3.0.0           # FastMCP framework for MCP server implementation
```

#### **Database Connectivity**
```bash
sqlalchemy>=2.0.0,<3.0.0         # Database ORM and connection management
psycopg2-binary>=2.9.0,<3.0.0    # PostgreSQL database adapter
snowflake-sqlalchemy>=1.5.0,<2.0.0     # Snowflake SQLAlchemy dialect
snowflake-connector-python>=3.0.0,<4.0.0  # Snowflake Python connector
```

#### **Configuration & Environment**
```bash
pydantic>=2.0.0,<3.0.0           # Data validation and settings management
python-dotenv>=1.0.0,<2.0.0      # Environment variable loading from .env files
```

#### **Semantic Web & Ontology**
```bash
rdflib>=7.0.0,<8.0.0             # RDF graph creation and manipulation
owlrl>=6.0.0,<7.0.0              # OWL reasoning and validation
```

#### **Automatic Dependencies (installed with above)**

When you install the main dependencies, these will be automatically installed:

**Database & Connection**:
- `boto3`, `botocore` - AWS SDK (for Snowflake S3 integration)
- `cryptography` - Encryption and security functions
- `pyOpenSSL` - SSL/TLS support
- `cffi` - C Foreign Function Interface
- `asn1crypto` - ASN.1 parsing and encoding

**Data Processing**:
- `sortedcontainers` - Sorted list/dict implementations
- `platformdirs` - Platform-specific directory locations
- `filelock` - File locking utilities

**Network & Auth**:
- `requests` - HTTP library
- `urllib3` - HTTP client
- `certifi` - Certificate bundle
- `pyjwt` - JWT token handling

**Configuration**:
- `tomlkit` - TOML file parsing
- `typing_extensions` - Enhanced type hints

### Manual Installation (if needed)

If you encounter issues with automatic installation, install key components manually:

```bash
# Core framework
pip install fastmcp>=2.10.0

# Database support
pip install sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0

# Snowflake support (may require additional system dependencies)
pip install snowflake-sqlalchemy snowflake-connector-python

# Semantic web
pip install rdflib>=7.0.0 owlrl>=6.0.0

# Configuration
pip install pydantic>=2.0.0 python-dotenv>=1.0.0
```

### System Dependencies

For some libraries, you might need system-level dependencies:

**macOS (via Homebrew)**:
```bash
brew install postgresql  # For psycopg2
brew install openssl     # For cryptographic functions
```

**Ubuntu/Debian**:
```bash
sudo apt-get install libpq-dev python3-dev  # For psycopg2
sudo apt-get install libssl-dev libffi-dev   # For cryptographic functions
```

**Windows**:
- Most dependencies work out of the box with pip
- For PostgreSQL support, ensure PostgreSQL client libraries are installed

## 🏗️ Streamlined Project Structure

```
database-ontology-mcp/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── main.py                     # Streamlined FastMCP server (8 tools, 623 lines)
│   ├── main_original.py            # Backup of previous complex version (2143 lines)
│   ├── main_backup.py              # Additional backup
│   ├── database_manager.py         # Database connection and analysis
│   ├── ontology_generator.py       # RDF ontology generation with SQL mappings
│   ├── config.py                   # Configuration management with .env support
│   ├── constants.py                # Application constants and settings
│   └── utils.py                    # Utility functions
├── run_server.py                   # Server startup script
├── .env                            # Environment configuration (DO NOT COMMIT)
├── requirements.txt                # Python dependencies
└── README.md                       # This comprehensive guide
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- PostgreSQL or Snowflake database access

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/database-ontology-mcp
cd database-ontology-mcp
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Note**: The charting functionality requires visualization libraries (pandas, plotly, matplotlib, seaborn). These are included in `requirements.txt` and will be installed automatically in your virtual environment.

4. **Configure environment:**
```bash
# Create .env file with your database credentials
cp .env.template .env  # If template exists, or create new .env
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# =================================================================
# Database Ontology MCP Server Configuration
# =================================================================

# Server Configuration
LOG_LEVEL=INFO
ONTOLOGY_BASE_URI=http://example.com/ontology/

# PostgreSQL Configuration (optional - can provide via tool parameters)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=postgres
POSTGRES_USERNAME=postgres
POSTGRES_PASSWORD=postgres

# Snowflake Configuration (optional - can provide via tool parameters)
SNOWFLAKE_ACCOUNT=CLYKFLK-KA74251    # Use your actual account identifier
SNOWFLAKE_USERNAME=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=SNOWFLAKE_SAMPLE_DATA
SNOWFLAKE_SCHEMA=TPCH_SF10
SNOWFLAKE_ROLE=PUBLIC

# Snowflake Troubleshooting:
# - Account format: Check Snowflake web UI URL for correct format
#   Common formats: CLYKFLK-KA74251, account.region, account.region.cloud
# - Role: Ensure your user has access to the specified role
# - Warehouse: Must be running and accessible
# - Database/Schema: Check permissions and case sensitivity
```

### Running the Server

**Make sure your virtual environment is activated:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python run_server.py
```

The server will start with streamlined tool output:

```
================================================================
Database Ontology MCP Server - Streamlined Edition
8 Essential Tools for Maximum Effectiveness
================================================================

🔧 Streamlined MCP Tools (8 Essential):
  1. connect_database      - Connect to PostgreSQL/Snowflake (uses .env fallback)
  2. list_schemas         - List available database schemas
  3. get_analysis_context - 🌟 MAIN TOOL: Complete analysis + automatic ontology
  4. generate_ontology    - Manual ontology generation (fallback)
  5. sample_table_data    - Secure data sampling with validation
  6. validate_sql_syntax  - SQL validation with fan-trap detection
  7. execute_sql_query    - Safe SQL execution with comprehensive warnings
  8. get_server_info      - Server status and configuration

🎯 Key Improvement: Main analysis tool now includes automatic ontology generation!
📊 Result: Claude Desktop gets semantic context automatically, no manual ontology steps needed.
🔒 Safety: Comprehensive fan-trap detection prevents query result multiplication.

🚀 Starting MCP server with HTTP streamable transport...
📡 Server ready and listening on port 8123 for HTTP MCP protocol messages
```

## 🔗 Claude Desktop Integration

### Option 1: Stdio Transport (Standard - Works for most features)

Add to your Claude Desktop MCP settings (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "database-ontology": {
      "command": "python",
      "args": ["/absolute/path/to/database-ontology-mcp/run_server.py"]
    }
  }
}
```

**Note**: Replace `/absolute/path/to/database-ontology-mcp/` with your actual project path.

### Option 2: HTTP Transport (For Chart Images - Experimental)

1. **Enable HTTP transport** by adding to your `.env` file:
```bash
MCP_USE_HTTP=true
```

2. **Add to Claude Desktop MCP settings**:
```json
{
  "mcpServers": {
    "database-ontology": {
      "url": "http://localhost:8123/mcp"
    }
  }
}
```

3. **Start the server manually**:
```bash
cd /path/to/database-ontology-mcp
source venv/bin/activate  # On Windows: venv\Scripts\activate
python run_server.py
```

### Recommendation

**Start with Option 1 (stdio transport)** - this is the standard way Claude Desktop connects to MCP servers and should work reliably. Chart images may be returned as file paths instead of being displayed directly.

## 🛠️ Streamlined MCP Tools Reference

### 🌟 Main Tool: `get_analysis_context`

The primary tool that provides complete database analysis with automatic ontology generation.

**Purpose**: Single tool that gives Claude Desktop everything it needs for database analysis.

**Parameters:**
```typescript
{
  schema_name?: string,     // Optional, defaults to public/default schema
  base_uri?: string,       // Optional, uses ONTOLOGY_BASE_URI from .env
  include_sample_data?: boolean  // Optional, includes sample data for context
}
```

**What it includes automatically:**
- Complete schema analysis (tables, columns, types, constraints)
- Relationship mapping with cardinality detection
- **Automatic ontology generation** with SQL references
- Data sampling for context (when requested)
- Fan-trap detection and prevention guidance
- Business context inference

**Returns:**
```typescript
{
  schema_analysis: {
    schema: string,
    tables: Array<TableInfo>,
    relationships: Array<RelationshipInfo>,
    fan_trap_warnings: Array<string>
  },
  ontology: string,  // RDF/Turtle format with db:sqlReference annotations
  sample_data?: Array<TableSamples>,
  recommendations: {
    safe_query_patterns: Array<string>,
    risk_warnings: Array<string>
  }
}
```

### Core Database Tools

#### 1. `connect_database`
Connect to PostgreSQL or Snowflake with environment variable fallback.

**Key Feature**: Parameters are optional - uses .env values when not provided.

**Parameters:**
```typescript
{
  db_type: "postgresql" | "snowflake",
  // All other parameters optional - falls back to .env values
  host?: string, port?: number, database?: string,
  username?: string, password?: string,
  account?: string, warehouse?: string, schema?: string, role?: string
}
```

**Examples:**
```python
# Simple connection using .env values
connect_database("postgresql")
connect_database("snowflake")

# Override specific parameters
connect_database("postgresql", host="custom.host.com", port=5433)
connect_database("snowflake", account="CUSTOM-ACCOUNT", warehouse="ANALYTICS_WH")
```

#### 2. `list_schemas`
Get available database schemas.

**Returns:** `Array<string>` of schema names

#### 3. `generate_ontology` (Fallback Tool)
Manual ontology generation when you need specific control.

**Parameters:**
```typescript
{
  schema_name?: string,
  base_uri?: string,
  include_business_context?: boolean
}
```

**Returns:** RDF ontology in Turtle format with embedded SQL references

### Data & Validation Tools

#### 4. `sample_table_data`
Secure data sampling with comprehensive validation.

**Parameters:**
```typescript
{
  table_name: string,       // Required, validated against SQL injection
  schema_name?: string,     // Optional schema specification
  limit?: number           // Max 1000, default 10
}
```

#### 5. `validate_sql_syntax`
Advanced SQL validation with fan-trap detection.

**Parameters:**
```typescript
{
  sql_query: string        // SQL to validate
}
```

**Returns:**
```typescript
{
  is_valid: boolean,
  syntax_errors: Array<string>,
  fan_trap_warnings: Array<string>,
  safety_score: number,
  recommendations: Array<string>
}
```

#### 6. `execute_sql_query`
Safe SQL execution with comprehensive safety protocols.

**Features:**
- **Fan-trap detection** - Prevents data multiplication errors
- **Query pattern analysis** - Identifies risky aggregation patterns
- **Result validation** - Checks if results make business sense
- **Execution limits** - Row limits and timeout protection

**Critical Safety Patterns Included:**
```sql
-- ✅ SAFE: UNION approach for multi-fact queries
WITH unified_facts AS (
    SELECT customer_id, sales_amount, 0 as returns FROM sales
    UNION ALL
    SELECT customer_id, 0, return_amount FROM returns
)
SELECT customer_id, SUM(sales_amount), SUM(returns) FROM unified_facts GROUP BY customer_id;

-- ❌ DANGEROUS: Direct joins with aggregation (causes fan-trap)
SELECT customer_id, SUM(sales_amount), SUM(return_amount)
FROM sales s LEFT JOIN returns r ON s.customer_id = r.customer_id
GROUP BY customer_id;  -- This multiplies sales_amount incorrectly!
```

#### 7. `get_server_info`
Comprehensive server status and configuration information.

**Returns:** Server version, available features, tool list, configuration details

## 🎯 Optimal Workflow for Claude Desktop

### Recommended Analytical Session Startup

Use this prompt to start an analytical session with automatic ontology integration:

```
I need to analyze the database. Please use get_analysis_context to provide a complete overview including schema analysis and ontology generation, then help me understand the data structure and suggest interesting analytical queries.
```

### Why This Works Better

**Before (Complex Version)**:
- 17+ tools caused confusion
- Claude Desktop ignored ontology generation
- Manual steps required for semantic context
- Tool dependencies created workflow issues

**Now (Streamlined Version)**:
- 8 essential tools, crystal clear purpose
- Main tool automatically includes ontology
- No manual ontology steps needed
- Self-contained functionality

## 🔒 Advanced Fan-Trap Protection

### The Fan-Trap Problem

Fan-traps occur when joining tables with 1:many relationships and using aggregation functions, causing data multiplication:

```sql
-- This query is WRONG and will inflate sales figures
SELECT c.customer_name, SUM(s.amount) as total_sales
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
LEFT JOIN shipments sh ON o.id = sh.order_id
GROUP BY c.customer_name;
-- If an order has multiple shipments, sales amount gets multiplied!
```

### Built-in Protection

Our tools provide automatic protection:

1. **Relationship Analysis** - Identifies all 1:many relationships
2. **Pattern Detection** - Flags dangerous query patterns
3. **Safe Alternatives** - Suggests UNION-based approaches
4. **Result Validation** - Checks if totals make sense

### Safe Query Patterns

The server promotes these proven patterns:

**UNION Approach (Recommended)**:
```sql
WITH unified_metrics AS (
    SELECT entity_id, sales_amount, 0 as shipped_qty, 'SALES' as metric_type FROM sales
    UNION ALL
    SELECT entity_id, 0, shipped_quantity, 'SHIPMENT' as metric_type FROM shipments
)
SELECT entity_id, SUM(sales_amount), SUM(shipped_qty) FROM unified_metrics GROUP BY entity_id;
```

## 🧪 Testing & Validation

### Quick Connection Test

```bash
# Test PostgreSQL connection
python3 -c "
from src.config import config_manager
from src.database_manager import DatabaseManager
db_config = config_manager.get_database_config()
db_manager = DatabaseManager()
success = db_manager.connect_postgresql(
    db_config.postgres_host, db_config.postgres_port,
    db_config.postgres_database, db_config.postgres_username, 
    db_config.postgres_password
)
print(f'PostgreSQL connection: {\"✅ Success\" if success else \"❌ Failed\"}')
"

# Test Snowflake connection  
python3 -c "
from src.config import config_manager
from src.database_manager import DatabaseManager
db_config = config_manager.get_database_config()
db_manager = DatabaseManager()
success = db_manager.connect_snowflake(
    db_config.snowflake_account, db_config.snowflake_username,
    db_config.snowflake_password, db_config.snowflake_warehouse,
    db_config.snowflake_database, db_config.snowflake_schema,
    db_config.snowflake_role
)
print(f'Snowflake connection: {\"✅ Success\" if success else \"❌ Failed\"}')
"
```

### Validate All Dependencies

```bash
# Check all required libraries are installed
python3 -c "
import sys
required_libs = [
    'fastmcp', 'sqlalchemy', 'psycopg2', 'snowflake.sqlalchemy', 
    'snowflake.connector', 'pydantic', 'dotenv', 'rdflib', 'owlrl'
]
missing = []
for lib in required_libs:
    try:
        __import__(lib)
        print(f'✅ {lib}')
    except ImportError:
        print(f'❌ {lib}')
        missing.append(lib)

if missing:
    print(f'\\nMissing libraries: {missing}')
    print('Run: pip install -r requirements.txt')
else:
    print('\\n🎉 All dependencies installed successfully!')
"
```

## 🔧 Configuration Troubleshooting

### Snowflake Connection Issues

**Account Format Problems**:
- Check your Snowflake web UI URL
- Common formats: `ORGNAME-ACCOUNT`, `ACCOUNT.REGION`, `ACCOUNT.REGION.CLOUD`
- Try: `CLYKFLK-KA74251`, `KA74251.us-east-1.aws`

**Role and Permissions**:
- Ensure user has access to specified role (default: PUBLIC)
- Verify warehouse is running and accessible
- Check database and schema permissions

### PostgreSQL Connection Issues

**Common Solutions**:
- Verify PostgreSQL service is running
- Check firewall/network connectivity
- Confirm credentials and database name
- Test with psql command line first

### Chart Creation Issues

**"Missing required visualization libraries" Error**:

This means the charting libraries aren't installed. **Solution**:

1. **Activate your virtual environment**:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install/reinstall requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, plotly, matplotlib, seaborn; print('✅ All chart libraries available')"
   ```

4. **Restart the MCP server** with the virtual environment activated

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Key areas for improvement:

1. **Additional Database Support** (MySQL, Oracle, etc.)
2. **Enhanced Ontology Patterns** (domain-specific templates)
3. **Advanced Query Safety** (more sophisticated fan-trap detection)
4. **Performance Optimization** (caching, connection pooling)

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Test with both PostgreSQL and Snowflake
4. Ensure all dependencies install correctly
5. Run code quality checks
6. Submit Pull Request

## 🔄 Version History

### Current - Streamlined Edition
- ✨ Reduced from 17+ tools to 8 essential tools (2143→623 lines in main.py)
- 🎯 Main tool with automatic ontology generation
- 🛡️ Advanced fan-trap detection and prevention
- 📦 Complete dependency management with installation guide
- 🔧 Environment variable fallback for all connection parameters
- ✅ Production-ready Snowflake support with account format handling

### Previous - Complex Edition (preserved in main_original.py)
- Multiple redundant tools causing confusion
- Manual ontology generation steps
- Tool bloat with 17+ separate functions
- Complex dependencies between tools

---

**🎯 Built for Maximum Effectiveness: One Main Tool + Automatic Ontology = Better SQL from Business Language**