# Database Ontology MCP Server v0.2.0

🚀 **Enhanced MCP server for database schema analysis and ontology generation with security, performance, and reliability improvements.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project provides a robust, production-ready Python-based MCP (Model Context Protocol) server using FastMCP that analyzes relational database schemas (PostgreSQL and Snowflake) and generates high-quality ontologies in RDF/Turtle format. The server includes comprehensive LLM integration capabilities for semantic ontology enrichment.

## 🌟 Key Features

### 🔗 Database Connectivity
- **PostgreSQL** and **Snowflake** support with connection pooling
- Enhanced connection management with retry logic and health checks
- Secure credential handling with environment variable validation
- Connection timeout and error recovery mechanisms

### 🔍 Advanced Schema Analysis
- Parallel table analysis for improved performance
- Comprehensive metadata extraction (columns, types, constraints, relationships)
- Foreign key relationship mapping and validation
- Row count estimation and data sampling with security controls

### 🎯 Intelligent Ontology Generation
- High-quality RDF/OWL ontology generation with proper validation
- Sophisticated SQL-to-XSD type mapping
- Cardinality constraints and relationship modeling
- Structured ontology serialization in multiple formats

### 🧠 LLM-Powered Enrichment
- Intelligent schema analysis and semantic naming suggestions
- Business domain context inference from table and column names
- Automated relationship labeling and description generation
- Extensible integration for advanced enrichment capabilities

### 🛡️ Security & Reliability
- Input validation and SQL injection prevention
- Credential sanitization in logs and error messages
- Comprehensive error handling with structured responses
- Rate limiting and connection pooling for production use

### ⚡ Performance & Scalability
- Concurrent table analysis using thread pools
- Efficient connection pooling and resource management
- Optimized memory usage for large schemas
- Configurable limits and timeouts

### 📊 Observability
- Structured JSON logging with configurable levels
- Comprehensive error tracking and metrics
- Performance timing and resource monitoring
- Health check endpoints and status reporting

## 🏗️ Enhanced Project Structure

```
database-ontology-mcp/
├── src/
│   ├── __init__.py                 # Package initialization and exports
│   ├── main.py                     # Enhanced FastMCP server application
│   ├── database_manager.py         # Advanced database connection and analysis
│   ├── ontology_generator.py       # Intelligent RDF ontology generation
│   ├── config.py                   # Configuration management system
│   ├── constants.py                # Application constants and settings
│   └── utils.py                    # Utility functions and helpers
├── tests/
│   ├── __init__.py
│   ├── test_server.py              # Comprehensive MCP server tests
│   ├── test_database_manager.py    # Database manager unit tests
│   └── test_ontology_generator.py  # Ontology generator tests
├── config/
│   └── .env.template               # Enhanced environment configuration
├── docs/                           # Documentation directory
├── pyproject.toml                  # Modern Python project configuration
├── requirements.txt                # Production dependencies
└── README.md                       # This comprehensive guide
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- PostgreSQL or Snowflake database access

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd database-ontology-mcp
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp config/.env.template .env
# Edit .env with your database credentials and preferences
```

### Configuration

The server uses environment variables for configuration. Copy `.env.template` to `.env` and customize:

```bash
# =================================================================
# Database Ontology MCP Server Configuration
# =================================================================

# Server Configuration
LOG_LEVEL=INFO                              # DEBUG, INFO, WARNING, ERROR
ONTOLOGY_BASE_URI=http://example.com/ontology/

# PostgreSQL Configuration (required for PostgreSQL connections)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=mydb
POSTGRES_USERNAME=user
POSTGRES_PASSWORD=password

# Snowflake Configuration (required for Snowflake connections)
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USERNAME=user
SNOWFLAKE_PASSWORD=password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=MYDB
SNOWFLAKE_SCHEMA=PUBLIC
```

### Running the Server

```bash
python run_server.py
```

The enhanced server will start with comprehensive logging and health checks:

```
================================================================
Database Ontology MCP Server v0.2.0
Enhanced with security, performance, and reliability improvements
================================================================

🔧 Available MCP Tools:
  • connect_database - Connect to PostgreSQL or Snowflake with enhanced security
  • list_schemas - List available database schemas
  • analyze_schema - Parallel analysis of tables and columns
  • generate_ontology - Generate RDF ontology with validation
  • sample_table_data - Sample table data with security controls
  • get_table_relationships - Analyze foreign key relationships
  • get_enrichment_data - Prepare data for LLM enrichment
  • apply_ontology_enrichment - Apply LLM suggestions to ontology
  • get_server_info - Get comprehensive server information

🗄️  Supported Databases: PostgreSQL, Snowflake
🧠  LLM Enrichment: Available via MCP prompts and tools
🔒  Security: Enhanced credential handling and input validation
⚡  Performance: Connection pooling and parallel processing
📊  Observability: Structured logging and comprehensive error handling

🚀 Starting MCP server...
📡 Server ready and listening for MCP protocol messages
```

## 🛠️ MCP Tools Reference

### Core Database Operations

#### `connect_database`
Establishes a secure connection to PostgreSQL or Snowflake with enhanced validation.

**Parameters:**
```typescript
{
  db_type: "postgresql" | "snowflake",
  host?: string,              // PostgreSQL only
  port?: number,              // PostgreSQL only
  database?: string,
  username?: string,
  password?: string,
  account?: string,           // Snowflake only
  warehouse?: string,         // Snowflake only
  schema?: string             // Optional, defaults by database type
}
```

**Enhanced Features:**
- Automatic parameter validation from environment variables
- Connection pooling with configurable pool sizes
- Health checks and automatic reconnection
- Secure credential handling

**Examples:**
```python
# PostgreSQL with connection pooling
result = connect_database(
    db_type="postgresql",
    host="localhost",
    port=5432,
    database="ecommerce_db",
    username="analytics_user",
    password="secure_password"
)

# Snowflake with warehouse optimization
result = connect_database(
    db_type="snowflake",
    account="your-org-account",
    username="analyst",
    password="secure_password",
    warehouse="ANALYTICS_WH",
    database="PROD_DB",
    schema="SALES"
)
```

#### `analyze_schema`
Performs comprehensive parallel analysis of database schema with enhanced performance.

**Parameters:**
```typescript
{
  schema_name?: string        // Optional, defaults to public/default schema
}
```

**Returns:**
```typescript
{
  schema: string,
  table_count: number,
  tables: Array<{
    name: string,
    schema: string,
    columns: Array<{
      name: string,
      data_type: string,
      is_nullable: boolean,
      is_primary_key: boolean,
      is_foreign_key: boolean,
      foreign_key_table?: string,
      foreign_key_column?: string,
      comment?: string
    }>,
    primary_keys: string[],
    foreign_keys: Array<{
      column: string,
      referenced_table: string,
      referenced_column: string
    }>,
    comment?: string,
    row_count?: number
  }>
}
```

#### `generate_ontology`
Generates a high-quality RDF/OWL ontology with optional LLM enrichment.

**Parameters:**
```typescript
{
  schema_name?: string,       // Schema to analyze
  base_uri?: string,          // Ontology base URI
  enrich_llm?: boolean        // Enable LLM enrichment
}
```

**Returns:** RDF ontology in Turtle format with proper validation

**Example Output:**
```turtle
@prefix ns: <http://company.com/ontology/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ns: a owl:Ontology ;
    rdfs:label "Database Schema Ontology" ;
    rdfs:comment "Ontology generated from database schema" .

ns:Customer a owl:Class ;
    rdfs:label "Customer" ;
    rdfs:comment "Customer entity representing registered users" .

ns:customerName a owl:DatatypeProperty ;
    rdfs:domain ns:Customer ;
    rdfs:range xsd:string ;
    rdfs:label "customerName" ;
    rdfs:comment "Full name of the customer" .

ns:belongsToCustomer a owl:ObjectProperty ;
    rdfs:domain ns:Order ;
    rdfs:range ns:Customer ;
    rdfs:label "belongsToCustomer" .
```

### Advanced LLM Integration

#### `get_enrichment_data`
Prepares structured data optimized for LLM-based ontology enrichment.

**Returns:**
```typescript
{
  schema_data: Array<{
    table_name: string,
    schema: string,
    columns: Array<ColumnInfo>,
    foreign_keys: Array<ForeignKey>,
    row_count?: number,
    comment?: string,
    sample_data?: Array<Record<string, any>>  // Limited samples for context
  }>,
  instructions: {
    task: string,
    expected_format: {
      classes: Array<ClassSuggestion>,
      properties: Array<PropertySuggestion>,
      relationships: Array<RelationshipSuggestion>
    },
    guidelines: string[]
  }
}
```

#### `apply_ontology_enrichment`
Applies LLM-generated enrichment suggestions to create a semantically rich ontology.

**Parameters:**
```typescript
{
  schema_name?: string,
  base_uri?: string,
  enrichment_suggestions: {
    classes: Array<{
      original_name: string,
      suggested_name: string,        // PascalCase
      description: string
    }>,
    properties: Array<{
      table_name: string,
      original_name: string,
      suggested_name: string,        // camelCase
      description: string
    }>,
    relationships: Array<{
      from_table: string,
      to_table: string,
      suggested_name: string,        // camelCase
      description: string
    }>
  }
}
```

### Utility Operations

#### `sample_table_data`
Securely samples table data with comprehensive validation.

**Parameters:**
```typescript
{
  table_name: string,         // Required, validated for security
  schema_name?: string,       // Optional schema specification
  limit?: number             // Max 1000, default 10
}
```

**Security Features:**
- SQL injection prevention
- Identifier validation
- Row limit enforcement
- Sensitive data handling

#### `get_table_relationships`
Analyzes and maps all foreign key relationships in a schema.

**Returns:**
```typescript
{
  [table_name: string]: Array<{
    column: string,
    referenced_table: string,
    referenced_column: string
  }>
}
```

#### `get_server_info`
Provides comprehensive server status and capability information.

**Returns:**
```typescript
{
  name: string,
  version: string,
  description: string,
  supported_databases: string[],
  features: string[],
  tools: string[],
  configuration: {
    log_level: string,
    base_uri: string,
    max_sample_limit: number,
    supported_formats: string[]
  }
}
```

## 🧪 Testing

The project includes comprehensive test coverage with multiple test suites:

### Running Tests

```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test suites
python -m pytest tests/test_database_manager.py -v
python -m pytest tests/test_ontology_generator.py -v
python -m pytest tests/test_server.py -v

# Run with detailed output
python -m unittest tests.test_server -v
```

### Test Structure

```
tests/
├── test_server.py              # MCP server and integration tests
├── test_database_manager.py    # Database connectivity and analysis tests
└── test_ontology_generator.py  # Ontology generation and enrichment tests
```

### Test Coverage Areas

- ✅ Database connection management and pooling
- ✅ Schema analysis and table introspection
- ✅ Ontology generation with various data types
- ✅ LLM enrichment workflow and validation
- ✅ Error handling and edge cases
- ✅ Security validation and input sanitization
- ✅ Configuration management
- ✅ Utility functions and helpers

## 🔧 Development

### Code Quality Tools

The project uses modern Python development tools:

```bash
# Code formatting
black src/ tests/

# Import sorting
isort src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Pre-commit hooks (recommended)
pre-commit install
```

### Adding New Database Support

1. **Extend DatabaseManager:**
```python
def connect_mysql(self, host: str, port: int, database: str, 
                  username: str, password: str) -> bool:
    """Connect to MySQL database."""
    # Implementation here
    pass
```

2. **Update constants:**
```python
SUPPORTED_DB_TYPES = ["postgresql", "snowflake", "mysql"]
```

3. **Add configuration validation:**
```python
def validate_db_config(self, db_type: str) -> Dict[str, Any]:
    # Add MySQL validation logic
    pass
```

### Extending Ontology Generation

The `OntologyGenerator` class supports various enhancements:

```python
def add_domain_specific_annotations(self, domain: str):
    """Add domain-specific OWL annotations."""
    pass

def generate_class_hierarchy(self, tables_info: List[TableInfo]):
    """Generate sophisticated class hierarchies."""
    pass

def add_inference_rules(self, rules: List[str]):
    """Add SWRL inference rules to the ontology."""
    pass
```

### Performance Optimization

Key areas for performance tuning:

- **Connection Pooling:** Adjust pool sizes in `DatabaseManager`
- **Parallel Processing:** Configure thread pool sizes in `ServerState`
- **Caching:** Implement schema metadata caching
- **Memory Management:** Optimize large table handling

## 📚 API Documentation

### MCP Protocol Integration

The server implements the Model Context Protocol (MCP) specification with enhanced features:

- **Structured Error Responses:** Consistent error handling with proper error types
- **Parameter Validation:** Comprehensive input validation and sanitization
- **Async Support:** Ready for asynchronous operation modes
- **Protocol Versioning:** Compatible with MCP v1.0+ specifications

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO | No |
| `ONTOLOGY_BASE_URI` | Base URI for generated ontologies | http://example.com/ontology/ | No |
| `POSTGRES_HOST` | PostgreSQL host | None | For PostgreSQL |
| `POSTGRES_PORT` | PostgreSQL port | 5432 | For PostgreSQL |
| `POSTGRES_DATABASE` | PostgreSQL database name | None | For PostgreSQL |
| `POSTGRES_USERNAME` | PostgreSQL username | None | For PostgreSQL |
| `POSTGRES_PASSWORD` | PostgreSQL password | None | For PostgreSQL |
| `SNOWFLAKE_ACCOUNT` | Snowflake account identifier | None | For Snowflake |
| `SNOWFLAKE_USERNAME` | Snowflake username | None | For Snowflake |
| `SNOWFLAKE_PASSWORD` | Snowflake password | None | For Snowflake |
| `SNOWFLAKE_WAREHOUSE` | Snowflake warehouse | None | For Snowflake |
| `SNOWFLAKE_DATABASE` | Snowflake database | None | For Snowflake |
| `SNOWFLAKE_SCHEMA` | Snowflake schema | PUBLIC | For Snowflake |

## 🔒 Security Considerations

### Production Deployment

- **Environment Variables:** Use secure secret management systems
- **Network Security:** Deploy behind firewalls and use VPNs
- **Authentication:** Implement proper database user permissions
- **Logging:** Ensure logs don't contain sensitive information
- **Rate Limiting:** Configure appropriate request limits
- **Connection Limits:** Set reasonable connection pool sizes

### Security Features

- ✅ SQL injection prevention through parameterized queries
- ✅ Input validation and sanitization
- ✅ Credential redaction in logs
- ✅ Connection timeout and retry limits
- ✅ Resource usage monitoring
- ✅ Error message sanitization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest tests/ -v`)
5. Run code quality checks (`black`, `isort`, `flake8`, `mypy`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📞 Support

- **Documentation:** Check this README and inline code documentation
- **Issues:** Report bugs and request features via GitHub Issues
- **Discussions:** Join community discussions for questions and ideas

## 🔄 Changelog

### v0.2.0 - Enhanced Production Release
- ✨ Complete architecture overhaul with enhanced security
- ⚡ Performance improvements with connection pooling and parallel processing
- 🛡️ Advanced error handling and input validation
- 📊 Comprehensive logging and observability
- 🧪 Extensive test coverage and quality assurance
- 📚 Complete documentation overhaul
- 🔧 Modern Python project configuration

### v0.1.0 - Initial Release
- Basic MCP server functionality
- PostgreSQL and Snowflake support
- Simple ontology generation
- Core schema analysis features

---

**Built with ❤️ for the database and semantic web communities**