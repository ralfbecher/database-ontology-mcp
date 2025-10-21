# Orionbelt Semantic Layer

**Orionbelt Semantic Layer - the Ontology-based MCP server for your Text-2-SQL convenience.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.12+-blue)](https://github.com/jlowin/fastmcp)

This project provides a production-ready Python-based MCP (Model Context Protocol) server that analyzes relational database schemas (PostgreSQL, Snowflake, and Dremio) and automatically generates comprehensive ontologies in RDF/Turtle format with direct SQL mappings.

## Key Philosophy: Automatic Ontology Integration

Our main analysis tool `get_analysis_context()` automatically includes ontology generation, making semantic context readily available for every query.

## 🌟 Key Features

### 🔗 Database Connectivity

- **PostgreSQL**, **Snowflake**, and **Dremio** support with connection pooling
- **Environment variable fallback** - parameters optional, uses .env when not provided
- **Enhanced connection management** with retry logic and timeout handling
- **Automatic dependency management** for Snowflake and Dremio connectors

### 🎯 9 Essential Tools

- **Streamlined workflow** with focused, purpose-built tools
- **Interactive charting** (`generate_chart`) with direct image rendering
- **Comprehensive schema analysis** with automatic ontology generation
- **Built-in workflow guidance** via FastMCP Context integration
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

## Python Library Installation

### Required Dependencies

```bash
# Install all required dependencies
uv sync
```

### Complete Library List

The project uses the following Python libraries:

#### **Core MCP Framework**

```bash
fastmcp>=2.12.0                  # FastMCP framework for MCP server implementation
```

#### **Database Connectivity**

```bash
sqlalchemy>=2.0.0,<3.0.0         # Database ORM and connection management
psycopg2-binary>=2.9.0,<3.0.0    # PostgreSQL database adapter
snowflake-sqlalchemy>=1.5.0,<2.0.0     # Snowflake SQLAlchemy dialect
snowflake-connector-python>=3.0.0,<4.0.0  # Snowflake Python connector
# Dremio uses PostgreSQL wire protocol (psycopg2-binary above)
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
pip install fastmcp>=2.12.0

# Database support
pip install sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0

# Snowflake support (may require additional system dependencies)
pip install snowflake-sqlalchemy snowflake-connector-python

# Dremio support (uses PostgreSQL protocol, psycopg2-binary already installed above)

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

## Project Structure

```
database-ontology-mcp/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── main.py                     # FastMCP server entry point (9 tools)
│   ├── database_manager.py         # Database connection and analysis
│   ├── ontology_generator.py       # RDF ontology generation with SQL mappings
│   ├── security.py                 # Security and validation utilities
│   ├── chart_utils.py              # Chart generation utilities
│   ├── config.py                   # Configuration management with .env support
│   ├── constants.py                # Application constants and settings
│   ├── utils.py                    # Utility functions
│   └── tools/                      # Tool implementations
│       ├── chart.py                # Chart generation tool
│       └── schema.py               # Schema analysis tools
├── tests/                          # Test suite
├── tmp/                            # Generated files (ontologies, charts)
├── server.py                       # Server startup script
├── .env                            # Environment configuration (DO NOT COMMIT)
├── pyproject.toml                  # Project metadata and dependencies
└── README.md                       # This comprehensive guide
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- PostgreSQL, Snowflake, or Dremio database access

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
uv sync
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

# Dremio Configuration (optional - can provide via tool parameters)
DREMIO_HOST=localhost
DREMIO_PORT=31010
DREMIO_USERNAME=your_username
DREMIO_PASSWORD=your_password

# Snowflake Troubleshooting:
# - Account format: Check Snowflake web UI URL for correct format
#   Common formats: CLYKFLK-KA74251, account.region, account.region.cloud
# - Role: Ensure your user has access to the specified role
# - Warehouse: Must be running and accessible
# - Database/Schema: Check permissions and case sensitivity

# Dremio Troubleshooting:
# - Host: Dremio coordinator node hostname or IP
# - Port: Default PostgreSQL wire protocol port is 31010
# - SSL: Enable/disable SSL connections (default: enabled)
# - Connection: Uses PostgreSQL protocol, no additional drivers needed
```

### Running the Server

**Make sure your virtual environment is activated:**

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python server.py
```

## Claude Desktop Integration

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

**Start the server manually**:

```bash
cd /path/to/database-ontology-mcp
source venv/bin/activate  # On Windows: venv\Scripts\activate
python server.py
```

## MCP Tools Reference

### Workflow Guidance

The server provides **built-in workflow guidance** through FastMCP Context integration, automatically suggesting the next recommended tool after each operation. This helps Claude Desktop users follow optimal analytical workflows without confusion.

**Key Workflows:**

1. **Complete Schema Analysis → Ontology → SQL**
   - `connect_database` → `analyze_schema` → `generate_ontology` → `execute_sql_query`

2. **Quick Data Exploration**
   - `connect_database` → `list_schemas` → `sample_table_data`

3. **SQL Validation → Execution → Visualization**
   - `validate_sql_syntax` → `execute_sql_query` → `generate_chart`

4. **Relationship Analysis for Complex Queries**
   - `analyze_schema` (check FKs) → `validate_sql_syntax` → `execute_sql_query`

### Core Database Tools

#### 1. `connect_database`

Connect to PostgreSQL, Snowflake, or Dremio with environment variable fallback.

**Key Feature**: Parameters are optional - uses .env values when not provided.

**Parameters:**

```typescript
{
  db_type: "postgresql" | "snowflake" | "dremio",
  // All other parameters optional - falls back to .env values
  host?: string, port?: number, database?: string,
  username?: string, password?: string,
  account?: string, warehouse?: string, schema?: string, role?: string,
  ssl?: boolean  // Dremio only
}
```

**Examples:**

```python
# Simple connection using .env values
connect_database("postgresql")
connect_database("snowflake")
connect_database("dremio")

# Override specific parameters
connect_database("postgresql", host="custom.host.com", port=5433)
connect_database("snowflake", account="CUSTOM-ACCOUNT", warehouse="ANALYTICS_WH")
connect_database("dremio", host="dremio.company.com", port=31010, ssl=False)
```

#### 2. `list_schemas`

Get available database schemas.

**Returns:** `Array<string>` of schema names

#### 3. `analyze_schema`

Analyze database schema and return comprehensive table information including relationships.

**Parameters:**
- `schema_name` (optional): Name of schema to analyze

**Returns:** Schema structure with tables, columns, primary keys, foreign keys, and relationship information

**Key Feature:** Foreign key analysis is critical for preventing fan-traps in SQL queries

#### 4. `generate_ontology`

Generate RDF/OWL ontology from database schema with SQL mapping annotations.

**Parameters:**
- `schema_info` (optional): JSON string with schema information
- `schema_name` (optional): Name of schema to generate ontology from
- `base_uri` (optional): Base URI for ontology (default: http://example.com/ontology/)

**Returns:** RDF ontology in Turtle format with `db:` namespace annotations

**Output:** Ontology is saved to `tmp/ontology_{schema}_{timestamp}.ttl`

### Data & Validation Tools

#### 5. `sample_table_data`

Secure data sampling with comprehensive validation.

**Parameters:**

```typescript
{
  table_name: string,       // Required, validated against SQL injection
  schema_name?: string,     // Optional schema specification
  limit?: number           // Max 1000, default 10
}
```

#### 6. `validate_sql_syntax`

Advanced SQL validation with comprehensive analysis.

**Parameters:**
- `sql_query` (required): SQL query to validate

**Returns:**
- `is_valid`: Boolean validation result
- `database_dialect`: Detected database dialect
- `validation_results`: Detailed component analysis
- `suggestions`: Optimization recommendations
- `warnings`: Performance concerns
- `errors`: Specific syntax errors
- `security_analysis`: Security findings

**Features:** Multi-database syntax checking, injection detection, performance analysis

#### 7. `execute_sql_query`

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

#### 8. `generate_chart`

Generate interactive charts from SQL query results.

**Parameters:**
- `data_source` (required): List of dictionaries (typically from `execute_sql_query`)
- `chart_type` (required): 'bar', 'line', 'scatter', or 'heatmap'
- `x_column` (required): Column name for X-axis
- `y_column` (optional): Column name for Y-axis
- `color_column` (optional): Column for color grouping
- `title` (optional): Chart title (auto-generated if not provided)
- `chart_library` (optional): 'matplotlib' or 'plotly' (default: matplotlib)
- `chart_style` (optional): 'grouped' or 'stacked' for bar charts
- `width` (optional): Chart width in pixels (default: 800)
- `height` (optional): Chart height in pixels (default: 600)

**Returns:** FastMCP Image object for direct display in Claude Desktop

**Output:** Chart saved to `tmp/chart_{timestamp}.png`

**Key Feature:** Direct image rendering without base64 encoding for better performance

#### 9. `get_server_info`

Comprehensive server status and configuration information.

**Returns:** Server version, available features, tool list, configuration details

## 🎯 Optimal Workflow for Claude Desktop

### Recommended Analytical Session Startup

The server provides **built-in comprehensive instructions** that are automatically sent to Claude Desktop, guiding optimal tool usage and workflows. This eliminates confusion and ensures accurate Text-to-SQL generation with fan-trap prevention.

**Recommended Starting Prompts:**

```
"Connect to my PostgreSQL database and analyze the schema with ontology generation"
```

```
"I need to query my Snowflake data warehouse - help me understand the schema relationships first"
```

### Key Improvements in Recent Updates

**FastMCP 2.12+ Integration**:
- Updated to latest FastMCP version with new resource API
- Removed deprecated `@mcp.list_resources()` and `@mcp.read_resource()` decorators
- Implemented new `@mcp.resource()` decorator with URI templates

**Chart Generation Enhancement**:
- Simplified to return Image objects directly (no resource URIs)
- Removed in-memory image store complexity
- Direct image rendering for better Claude Desktop integration
- Charts saved to `tmp/` directory for reference

**Workflow Guidance**:
- Added FastMCP Context parameter to all 9 tools
- Automatic next-tool suggestions after each operation
- Comprehensive server instructions for optimal workflows
- Built-in fan-trap prevention guidance

## Fan-Trap Protection

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

## Testing & Validation

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

## 🧪 Testing & Quality

The project includes a comprehensive test suite with significant improvements in recent updates:

**Test Coverage:**
- 70% pass rate with 24 failures (down from 42)
- Fixed 18 tests across Ontology Generator, Security, and Database Manager
- Comprehensive test coverage for core functionality

**Security Testing:**
- SQL injection pattern detection (including comment-based attacks)
- Identifier sanitization validation
- Credential handling security

**Test Improvements:**
- Enhanced mock setups for database connections
- Better test isolation and cleanup
- Documented remaining test issues for future work

**Running Tests:**

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_database_manager.py
```

## Configuration Troubleshooting

### Snowflake Connection Issues

**Account Format Problems**:

- Check your Snowflake web UI URL
- Account format: `ORGNAME-ACCOUNT`

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📋 Recent Changes

### Version 0.3.0

**FastMCP 2.12+ Upgrade** (Oct 2025):
- Upgraded to FastMCP 2.12+ with new resource API
- Replaced deprecated resource decorators
- Fixed resource API deprecation warnings

**Chart Generation Simplification** (Oct 2025):
- Return Image objects directly instead of resource URIs
- Removed in-memory image store complexity
- Simplified error handling with exceptions
- Better Claude Desktop integration

**Workflow Guidance Enhancement** (Oct 2025):
- Added FastMCP Context parameter to all tools
- Automatic next-tool suggestions
- Comprehensive MCP server instructions
- Improved client workflow guidance

**Test & Security Improvements** (Oct 2025):
- 43% reduction in test failures (18 tests fixed)
- Enhanced SQL injection detection (comment-based attacks)
- Improved identifier sanitization
- Better ontology generation with enrichment methods

## Contributing

We welcome contributions! Key areas for improvement:

1. **Additional Database Support** (MySQL, Oracle, etc.)
2. **Enhanced Ontology Patterns** (domain-specific templates)
3. **Advanced Query Safety** (more sophisticated fan-trap detection)
4. **Performance Optimization** (caching, connection pooling)
5. **Test Coverage** (increase from current 70% pass rate)

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Test with both PostgreSQL and Snowflake
4. Ensure all dependencies install correctly
5. Run code quality checks (`pytest`, `black`, `flake8`)
6. Submit Pull Request

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Run tests
pytest

# Format code
black src/ tests/

# Run linting
flake8 src/ tests/

# Type checking
mypy src/
```

---
