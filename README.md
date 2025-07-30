# Database Ontology MCP Server

This project provides a Python-based MCP (Model Context Protocol) server using FastMCP that analyzes relational database schemas (PostgreSQL and Snowflake) and generates ontologies in RDF/Turtle format. The server includes placeholders for future LLM integration to enrich the generated ontologies.

## Features

- **Database Connectivity**: Support for PostgreSQL and Snowflake databases
- **Schema Analysis**: Extract table structures, columns, data types, and relationships
- **Ontology Generation**: Convert database schemas to RDF/OWL ontologies in Turtle format
- **MCP Protocol**: Compatible with Model Context Protocol for AI integration
- **Extensible Design**: Ready for LLM-based ontology enrichment

## Project Structure

```
database-ontology-mcp/
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastMCP server application
│   ├── database_manager.py     # Database connection and analysis
│   └── ontology_generator.py   # RDF ontology generation
├── tests/
│   ├── __init__.py
│   └── test_server.py         # Unit tests
├── config/
│   └── .env.template          # Environment variables template
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone or create the project directory:
```bash
mkdir database-ontology-mcp
cd database-ontology-mcp
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Copy and configure environment variables:
```bash
cp config/.env.template .env
# Edit .env with your database credentials
```

## Configuration

Edit the `.env` file with your database connection details:

### PostgreSQL Configuration
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=mydb
POSTGRES_USERNAME=user
POSTGRES_PASSWORD=password
```

### Snowflake Configuration
```env
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USERNAME=user
SNOWFLAKE_PASSWORD=password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=MYDB
SNOWFLAKE_SCHEMA=PUBLIC
```

## Usage

### Running the MCP Server

```bash
python run_server.py
```

Or alternatively:
```bash
python -m src.main
```

The server will start and expose MCP tools for database analysis and ontology generation.

### Available MCP Tools

The server provides the following MCP tools that can be called by MCP clients:

#### 1. connect_database
Connect to a PostgreSQL or Snowflake database.

**Parameters:**
- `db_type` (str): Database type - either 'postgresql' or 'snowflake'
- `host` (str, optional): Database host (for PostgreSQL)
- `port` (int, optional): Database port (for PostgreSQL)
- `database` (str, optional): Database name
- `username` (str, optional): Database username
- `password` (str, optional): Database password
- `account` (str, optional): Snowflake account identifier (for Snowflake)
- `warehouse` (str, optional): Snowflake warehouse (for Snowflake)
- `schema` (str, optional): Database schema (default: PUBLIC)

**Example Usage:**
```python
# PostgreSQL
result = connect_database(
    db_type="postgresql",
    host="localhost",
    port=5432,
    database="mydb",
    username="user",
    password="password"
)

# Snowflake
result = connect_database(
    db_type="snowflake",
    account="your-account",
    username="user",
    password="password",
    warehouse="COMPUTE_WH",
    database="MYDB",
    schema="PUBLIC"
)
```

#### 2. list_schemas
Get available schemas in the connected database.

**Returns:** List of schema names

#### 3. analyze_schema
Analyze tables and relationships in a specific schema.

**Parameters:**
- `schema_name` (str, optional): Name of the schema to analyze

**Returns:** Dictionary containing detailed table information

#### 4. generate_ontology
Generate an RDF ontology from the database schema.

**Parameters:**
- `schema_name` (str, optional): Name of the schema to generate ontology from
- `base_uri` (str): Base URI for the ontology (default: http://example.com/ontology/)
- `enrich_llm` (bool): Whether to enrich with LLM insights (default: False)

**Returns:** RDF ontology in Turtle format

#### 5. sample_table_data
Sample data from a specific table for analysis.

**Parameters:**
- `table_name` (str): Name of the table to sample
- `schema_name` (str, optional): Schema containing the table
- `limit` (int): Maximum number of rows to return (default: 10)

**Returns:** List of sample rows as dictionaries

#### 6. get_table_relationships
Get foreign key relationships between tables in a schema.

**Parameters:**
- `schema_name` (str, optional): Name of the schema to analyze relationships

**Returns:** Dictionary mapping table names to their foreign key relationships

## Example Ontology Output

```turtle
@prefix ex: <http://example.com/ontology/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:users a owl:Class ;
    rdfs:label "users" .

ex:users_id a owl:DatatypeProperty ;
    rdfs:domain ex:users ;
    rdfs:range rdfs:Literal .

ex:users_name a owl:DatatypeProperty ;
    rdfs:domain ex:users ;
    rdfs:range rdfs:Literal .
```

## Testing

Run the unit tests:

```bash
python -m pytest tests/
```

Or run specific tests:

```bash
python -m unittest tests.test_server
```

## Development

### Adding New Database Support

To add support for additional databases:

1. Extend `DatabaseManager` class in `database_manager.py`
2. Add new connection method (e.g., `connect_mysql()`)
3. Update the MCP handler in `main.py`

### Extending Ontology Generation

The `OntologyGenerator` class can be extended to:

- Add more sophisticated OWL constructs
- Implement better data type mappings
- Generate more complex class hierarchies
- Add domain-specific annotations

### LLM Integration

The `enrich_with_llm()` method in `OntologyGenerator` is a placeholder for future LLM integration. This could include:

- Generating semantic class names and descriptions
- Inferring conceptual relationships between entities  
- Adding domain-specific annotations
- Creating more sophisticated ontological structures

## Dependencies

- `fastmcp`: FastMCP framework for MCP server implementation
- `sqlalchemy`: Database connectivity and ORM
- `psycopg2-binary`: PostgreSQL adapter
- `snowflake-sqlalchemy`: Snowflake adapter
- `rdflib`: RDF graph manipulation and serialization
- `pydantic`: Data validation and serialization
- `python-dotenv`: Environment variable management

## License

This project is provided as-is for educational and development purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions, please check the documentation or create an issue in the project repository.
