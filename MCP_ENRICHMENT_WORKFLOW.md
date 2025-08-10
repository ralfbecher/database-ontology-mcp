# MCP-Based Ontology Enrichment Workflow

This document explains how to use the new MCP-based ontology enrichment system with MCP prompts and tools.

## Overview

The enrichment system now uses two main MCP tools working in tandem with Claude (or other MCP-compatible AI assistants):

1. **`get_enrichment_data`** - Extracts structured schema data for analysis
2. **`apply_ontology_enrichment`** - Applies enrichment suggestions to generate an improved ontology

## Workflow Steps

### 1. Connect to Database

First, connect to your database using the standard connection tool:

```
Use the connect_database tool with your database credentials
```

### 2. Get Enrichment Data

Extract structured schema information for analysis:

```
Use get_enrichment_data tool with optional schema_name parameter
```

This returns:

- Detailed table and column information
- Sample data for context
- Structured format guidelines
- Enrichment instructions

### 3. Analyze and Create Suggestions

Use the enrichment data with MCP prompts to analyze the schema:

```
Use the ontology_enrichment_guide prompt for general guidance
Use the ontology_analysis_prompt with the schema data for specific analysis
```

The AI assistant will analyze the schema and provide enrichment suggestions in the required JSON format:

```json
{
  "classes": [
    {
      "original_name": "users",
      "suggested_name": "User",
      "description": "Represents a system user with authentication and profile information"
    }
  ],
  "properties": [
    {
      "table_name": "users",
      "original_name": "created_at",
      "suggested_name": "registrationDateTime",
      "description": "Timestamp when the user account was created"
    }
  ],
  "relationships": [
    {
      "from_table": "orders",
      "to_table": "users",
      "suggested_name": "belongsToUser",
      "description": "Links each order to the user who placed it"
    }
  ]
}
```

### 4. Apply Enrichment

Use the suggestions to generate an enriched ontology:

```
Use apply_ontology_enrichment tool with:
- schema_name (optional)
- base_uri (optional, defaults to http://example.com/ontology/)
- enrichment_suggestions (the JSON from step 3)
```

## Key Benefits

### No API Keys Required

- No need for external API keys or service dependencies
- Works with any MCP-compatible AI assistant (Claude, etc.)
- Fully integrated into the MCP protocol

### Interactive Process

- Can iterate on suggestions before applying
- Easy to modify or refine enrichment suggestions
- Transparent process where you can see and control each step

### Flexible and Extensible

- Can be used with different AI models through MCP
- Easy to customize prompts for specific domains
- Supports partial enrichment (e.g., only classes or only properties)

## Example Complete Workflow

1. **Connect**: `connect_database("postgresql", host="localhost", port=5432, ...)`
2. **Extract Data**: `get_enrichment_data("public")`
3. **Analyze**: Use prompts with Claude to analyze the schema data
4. **Generate Suggestions**: Claude provides enrichment JSON
5. **Apply**: `apply_ontology_enrichment(schema_name="public", enrichment_suggestions={...})`
6. **Result**: Receive enriched RDF ontology in Turtle format

## Advanced Usage

### Custom Domain Analysis

You can customize the analysis by providing domain-specific context in your prompts:

"Analyze this e-commerce database schema focusing on customer journey and order fulfillment processes..."

### Iterative Refinement

You can run multiple rounds of enrichment:

1. Get basic suggestions
2. Refine based on domain knowledge
3. Apply and evaluate results
4. Iterate with focused improvements

### Partial Enrichment

You can apply enrichment to specific parts of the schema:

- Only enrich core business entities
- Focus on most frequently used tables
- Enrich relationships separately from properties
