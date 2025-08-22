# Enriched Schema Context for Better SQL Generation

This document explains how to use the new **enriched schema context** functionality that combines database ontology data with schema information to help Claude Desktop generate better SQL from business language.

## What's New

The new `get_enriched_schema_context` tool provides a unified view that combines:

- **Raw schema information** (tables, columns, relationships) 
- **Semantic ontology mappings** (business-friendly names and descriptions)
- **SQL generation hints** (relationship warnings, join recommendations)
- **Sample data context** (when available)

## Key Benefits

### 1. Better SQL from Business Language
- Translates business terms to correct table/column names
- Understands semantic meaning behind database structures
- Provides context for generating accurate joins

### 2. Fan-Trap Prevention
- Automatically detects potential fan-trap scenarios
- Recommends safe aggregation patterns (UNION, separate CTEs)
- Warns about risky multi-table joins with aggregations

### 3. Semantic Understanding
- Maps technical column names to business concepts
- Provides intelligent descriptions based on naming patterns
- Helps Claude understand data purpose and usage

## Usage Examples

### 1. Get Basic Enriched Context

```python
# Connect to database first
result = connect_database("postgresql")

# Get enriched context for the default schema
context = get_enriched_schema_context()
```

### 2. Get Context for Specific Schema

```python
# Get context for a specific schema with full ontology
context = get_enriched_schema_context(
    schema_name="sales", 
    include_ontology=True,
    include_sample_data=True
)
```

### 3. Use Context for SQL Generation

```python
# Get the context
context = get_enriched_schema_context(schema_name="ecommerce")

# Use the sql_generation_context_prompt to get formatted context
from src.main import sql_generation_context_prompt
formatted_context = sql_generation_context_prompt(context)

# Now Claude has rich context for generating SQL from business requests like:
# "Show me total sales by customer for last quarter"
```

## Context Structure

The enriched context returns a dictionary with these key sections:

### 1. Schema Info
Raw database schema information from `analyze_schema`:
```json
{
  "schema_info": {
    "schema": "public",
    "table_count": 5,
    "tables": [
      {
        "name": "customers",
        "columns": [...],
        "primary_keys": [...],
        "foreign_keys": [...]
      }
    ]
  }
}
```

### 2. Semantic Mappings
Business-friendly interpretations of database structures:
```json
{
  "semantic_mappings": {
    "customers": {
      "business_name": "Customers",
      "description": "Customer information and profiles",
      "columns": {
        "customer_id": {
          "business_name": "Customer ID",
          "description": "Unique identifier for customer",
          "data_type": "INTEGER",
          "is_key": true
        },
        "customer_name": {
          "business_name": "Customer Name", 
          "description": "Name or title",
          "data_type": "VARCHAR(255)",
          "is_key": false
        }
      }
    }
  }
}
```

### 3. SQL Generation Hints
Warnings and recommendations for safe SQL generation:
```json
{
  "sql_generation_hints": {
    "relationship_warnings": [
      {
        "table": "orders",
        "warning": "Table orders has multiple foreign keys - potential fan-trap risk",
        "referenced_tables": ["customers", "order_items"],
        "recommendation": "Use separate CTEs or UNION approach for multi-fact queries"
      }
    ],
    "join_recommendations": [
      {
        "from_table": "orders",
        "to_table": "customers", 
        "join_condition": "orders.customer_id = customers.customer_id",
        "relationship_type": "many_to_one"
      }
    ]
  }
}
```

### 4. Relationships
Foreign key relationships between tables for proper joins.

### 5. Metadata
Information about the context generation (schema name, table count, etc.).

## Intelligent Column Descriptions

The system automatically generates intelligent descriptions based on column naming patterns:

| Pattern | Description |
|---------|-------------|
| `*_id` | "Unique identifier for [entity]" |
| `*name*` | "Name or title" |
| `*date*`, `*time*` | "Date/time information" |
| `*amount*`, `*price*`, `*cost*` | "Monetary or quantity value" |
| `*count*`, `*quantity*` | "Numeric count or quantity" |
| `*status*`, `*state*` | "Status or state indicator" |

## Integration with Existing Tools

### Workflow for Safe SQL Generation

1. **Connect**: `connect_database("postgresql")`
2. **Get Context**: `get_enriched_schema_context(schema_name="sales")`
3. **Understand Business Request**: Use semantic mappings to translate business terms
4. **Generate SQL**: Create SQL with proper joins and safe aggregation patterns
5. **Validate**: `validate_sql_syntax(sql_query)` 
6. **Execute**: `execute_sql_query(sql_query)`

### Using with Existing Prompts

The `sql_generation_context_prompt` creates a comprehensive context that Claude can use:

```python
context = get_enriched_schema_context()
prompt_text = sql_generation_context_prompt(context)
# Use prompt_text to inform Claude about the database structure and best practices
```

## Best Practices

### 1. Always Check for Fan-Traps
When the context includes relationship warnings:
- Use UNION approach for multi-fact queries
- Aggregate fact tables separately before joining
- Avoid direct joins between multiple fact tables

### 2. Use Semantic Mappings
- Translate business terms using the semantic mappings
- Understand the business purpose of each table and column
- Generate more meaningful SQL with proper aliases

### 3. Leverage Join Recommendations
- Use the provided join conditions for accurate relationships
- Follow the recommended relationship patterns
- Apply proper table aliases for readability

## Example: Business Language to SQL

**Business Request**: "Show me total sales and number of orders per customer"

**Using Enriched Context**:
1. Context shows `customers` table connects to `orders` table
2. Semantic mapping shows `order_total` is "Monetary value"
3. No relationship warnings for this simple aggregation
4. Recommended join: `orders.customer_id = customers.customer_id`

**Generated SQL**:
```sql
SELECT 
    c.customer_name,
    SUM(o.order_total) as total_sales,
    COUNT(o.order_id) as number_of_orders
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name
ORDER BY total_sales DESC;
```

## Advanced Usage

### Custom Ontology Enhancement
The enriched context can be combined with the existing ontology enrichment tools:

1. `get_enrichment_data()` - Get data for LLM analysis
2. `apply_ontology_enrichment()` - Apply custom semantic suggestions
3. `get_enriched_schema_context()` - Get the final enriched context

### Integration with External Systems
The enriched context is designed to be:
- JSON serializable for API integration
- Cacheable for performance
- Extensible for custom business rules
- Compatible with existing MCP tools

This enhancement makes the Database Ontology MCP Server significantly more powerful for natural language to SQL generation, while maintaining all existing functionality.