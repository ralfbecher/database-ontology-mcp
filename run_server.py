#!/usr/bin/env python3
"""Startup script for the Database Ontology MCP Server."""

import logging
import sys
import os
from src.main import mcp, cleanup_server

# Configure logging to stderr so it doesn't interfere with MCP protocol
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

def main():
    """Start the MCP server."""
    logger.info("Starting Database Ontology MCP Server v0.1.0...")
    logger.info("Available tools:")
    logger.info("  - connect_database: Connect to PostgreSQL or Snowflake")
    logger.info("  - list_schemas: List available database schemas")
    logger.info("  - analyze_schema: Analyze tables and columns in a schema")
    logger.info("  - generate_ontology: Generate RDF ontology from schema")
    logger.info("  - sample_table_data: Sample data from a specific table")
    logger.info("  - get_table_relationships: Get foreign key relationships")
    logger.info("  - get_server_info: Get server information and capabilities")
    logger.info("Server supports PostgreSQL and Snowflake databases")
    logger.info("LLM enrichment available with OPENAI_API_KEY environment variable")
    
    try:
        # Run the MCP server using stdio transport
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("Cleaning up server resources...")
        cleanup_server()

if __name__ == "__main__":
    main()
