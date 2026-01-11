# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Orionbelt Semantic Layer** - A FastMCP server that analyzes database schemas (PostgreSQL, Snowflake, Dremio) and generates RDF/OWL ontologies with embedded SQL mappings. Enables Text-to-SQL with fan-trap prevention and relationship-aware query construction.

## Build & Development Commands

```bash
# Install dependencies (Python 3.13+ required)
uv sync

# Start MCP server (default: localhost:9000)
uv run server.py

# Run tests with coverage
pytest

# Run a single test file
pytest tests/test_ontology_generator.py -v

# Run a specific test
pytest tests/test_ontology_generator.py::test_function_name -v

# Code formatting
black src/ tests/
isort src/ tests/

# Linting and type checking
flake8 src/
mypy src/

# Run all pre-commit checks
pre-commit run --all-files

# Security scan
bandit -r src/
```

## Architecture

### Core Components

- **`server.py`** - Entry point with startup logging and signal handling
- **`src/main.py`** (~2400 lines) - FastMCP server with 11 MCP tools as `@mcp.tool()` async decorators
- **`src/database_manager.py`** (~1900 lines) - Connection pooling, schema analysis, SQLAlchemy integration
- **`src/ontology_generator.py`** (~900 lines) - RDF/OWL generation using rdflib with `db:` namespace annotations
- **`src/r2rml_generator.py`** (~360 lines) - W3C R2RML mapping generation
- **`src/security.py`** (~350 lines) - SQL injection prevention, fan-trap detection, credential encryption
- **`src/chart_utils.py`** (~700 lines) - Plotly/Matplotlib chart generation

### Key Patterns

**Per-Session State Isolation**: Each MCP session maintains isolated `SessionData` with its own `DatabaseManager`, preventing cross-session interference:
```python
class SessionData:
    db_manager: DatabaseManager
    schema_file: Optional[Path]
    ontology_file: Optional[Path]
    r2rml_file: Optional[Path]
```

**Fan-Trap Prevention**: Multi-step validation prevents data multiplication errors:
1. `analyze_schema()` extracts FK relationships
2. Pattern detection in `execute_sql_query()`
3. Suggests UNION ALL patterns for multi-fact aggregation

**Ontology Triple Storage**: RDF graphs link back to SQL:
- `ns:TableName` → OWL:Class with `db:tableName`, `db:primaryKey`
- `ns:relationship` → OWL:ObjectProperty with `db:joinCondition`
- Uses `rdfs:comment` for business descriptions

### MCP Tools (in `src/main.py`)

`connect_database`, `list_schemas`, `analyze_schema`, `generate_ontology`, `suggest_semantic_names`, `apply_semantic_names`, `load_my_ontology`, `sample_table_data`, `validate_sql_syntax`, `execute_sql_query`, `generate_chart`, `get_server_info`

## Configuration

Key environment variables (see `.env.template`):
- `MCP_TRANSPORT` - `http` (default) or `sse`
- `MCP_SERVER_PORT` - Server port (default: 9000)
- `ONTOLOGY_BASE_URI` - RDF base URI
- `R2RML_BASE_IRI` - R2RML mapping IRI
- `OUTPUT_DIR` - Generated files directory (default: `tmp`)
- Database configs: `POSTGRES_*`, `SNOWFLAKE_*`, `DREMIO_*`

## Code Standards

- Line length: 88 characters (black)
- Type hints required for all public functions (mypy strict)
- Google-style docstrings
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

## Dependencies

Core: `fastmcp>=2.13.2`, `sqlalchemy>=2.0.36`, `rdflib>=7.1.0`, `pydantic>=2.10.0`

Visualization: `plotly>=5.24.0`, `matplotlib>=3.9.0`, `pandas>=2.2.0`
