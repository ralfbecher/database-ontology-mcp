#!/usr/bin/env python3
"""
Database Ontology MCP Server - Refactored

A focused MCP server with 11 essential tools for database analysis with automatic ontology generation and interactive charting.
Main tool: get_analysis_context() - provides complete schema analysis with integrated ontology.
"""

import logging
import shutil
from pathlib import Path

from fastmcp import FastMCP
from .config import config_manager
from . import __version__, __name__ as SERVER_NAME, __description__

# Import all tools from the tools package
from .tools import (
    connect_database,
    diagnose_connection_issue,
    list_schemas,
    get_analysis_context,
    sample_table_data,
    generate_ontology,
    load_ontology_from_file,
    validate_sql_syntax,
    execute_sql_query,
    generate_chart,
    get_server_info
)

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

# Register all tools with MCP
mcp.tool()(connect_database)
mcp.tool()(diagnose_connection_issue)
mcp.tool()(list_schemas)
mcp.tool()(get_analysis_context)
mcp.tool()(sample_table_data)
mcp.tool()(generate_ontology)
mcp.tool()(load_ontology_from_file)
mcp.tool()(validate_sql_syntax)
mcp.tool()(execute_sql_query)
mcp.tool()(generate_chart)
mcp.tool()(get_server_info)

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