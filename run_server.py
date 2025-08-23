#!/usr/bin/env python3
"""Startup script for the Database Ontology MCP Server."""

import sys
import os
import signal
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.main import mcp
from src.config import config_manager
from src.utils import setup_logging
from src import __version__, __name__ as SERVER_NAME

# Setup logging
config = config_manager.get_server_config()
logger = setup_logging(config.log_level, structured=False)  # Use simple format for startup

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def print_startup_info():
    """Print server startup information."""
    logger.info("="*60)
    logger.info(f"{SERVER_NAME} v{__version__}")
    logger.info("MCP server for database ad hoc analysis with ontology support and interactive charting")
    logger.info("="*60)
    
    logger.info("🔧 Available MCP Tools:")
    tools = [
        "connect_database - Connect to PostgreSQL or Snowflake with security",
        "list_schemas - List available database schemas",
        "analyze_schema - Parallel analysis of tables and columns", 
        "generate_ontology - Generate RDF ontology with validation",
        "sample_table_data - Sample table data with security controls",
        "get_table_relationships - Analyze foreign key relationships",
        "get_enrichment_data - Prepare data for LLM enrichment",
        "apply_ontology_enrichment - Apply LLM suggestions to ontology",
        "get_server_info - Get comprehensive server information"
    ]
    for tool in tools:
        logger.info(f"  • {tool}")
    
    logger.info("")
    logger.info("🗄️ Supported Databases: PostgreSQL, Snowflake")
    logger.info("🧠 LLM Enrichment: Available via MCP prompts and tools")
    logger.info("🔒 Security: Credential handling and input validation")
    logger.info("⚡ Performance: Connection pooling and parallel processing")
    logger.info("📊 Observability: Structured logging and comprehensive error handling")
    logger.info("")
    logger.info(f"📋 Configuration:")
    logger.info(f"  • Log Level: {config.log_level}")
    logger.info(f"  • Base URI: {config.ontology_base_uri}")
    logger.info(f"  • HTTP Server: {config.http_host}:{config.http_port}")
    logger.info("")

def main():
    """Start the enhanced MCP server."""    
    try:
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers()
        
        # Print startup information
        print_startup_info()
        
        # Check if we should use HTTP or stdio transport
        use_http = os.getenv("MCP_USE_HTTP", "false").lower() == "true"
        
        if use_http:
            logger.info("🚀 Starting MCP server with HTTP streamable transport...")
            logger.info(f"📡 Server ready and listening on {config.http_host}:{config.http_port}/mcp for HTTP MCP protocol messages")
            
            # Start the server with HTTP transport for better resource streaming
            mcp.run(transport="streamable-http", host=config.http_host, port=config.http_port, path="/mcp")
        else:
            logger.info("🚀 Starting MCP server with stdio transport...")
            logger.info("📡 Server ready for stdio MCP protocol messages")
            
            # Start the server with stdio transport (standard for Claude Desktop)
            mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Server stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Critical server error: {type(e).__name__}: {e}")
        logger.error("Please check your configuration and try again")
        return 1
    finally:
        logger.info("✅ Server shutdown complete")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
