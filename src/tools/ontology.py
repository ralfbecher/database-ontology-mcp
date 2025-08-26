"""Ontology generation and management tools."""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ..config import config_manager
from ..database_manager import DatabaseManager
from ..ontology_generator import OntologyGenerator

logger = logging.getLogger(__name__)

# Global database manager instance  
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def create_error_response(message: str, error_type: str, details: str = None) -> Dict[str, Any]:
    """Create a standardized error response."""
    error_response = {
        "success": False,
        "error": message,
        "error_type": error_type
    }
    if details:
        error_response["details"] = details
    return error_response


def generate_ontology(
    schema_name: Optional[str] = None,
    base_uri: Optional[str] = None,
    enrich_llm: bool = False
) -> str:
    """Generate a database ontology with direct SQL generation support.
    
    ℹ️  Most users should use get_analysis_context() instead, which includes ontology automatically.
    
    This tool generates a comprehensive ontology containing:
    - Direct database table/column references (customers.customer_id)
    - Ready-to-use JOIN conditions (orders.customer_id = customers.customer_id)
    - Business-friendly descriptions for understanding data meaning
    - Complete metadata (data types, constraints, row counts)
    
    Args:
        schema_name: Name of the schema to generate ontology from (optional)
        base_uri: Base URI for the ontology (optional, uses config default)
        enrich_llm: Whether to enrich the ontology with LLM insights (default: False)
    
    Returns:
        RDF ontology in Turtle format with complete database mappings
    """
    try:
        db_manager = get_db_manager()
        server_config = config_manager.get_server_config()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Please use connect_database tool first to establish a connection"
            )
        
        try:
            tables = db_manager.get_tables(schema_name)
            if not tables:
                return create_error_response(
                    f"No tables found in schema '{schema_name or 'default'}'",
                    "data_error"
                )
            
            # Analyze tables
            tables_info = []
            for table_name in tables:
                try:
                    table_info = db_manager.analyze_table(table_name, schema_name)
                    if table_info:
                        tables_info.append(table_info)
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table_name}: {e}")
            
            if not tables_info:
                return create_error_response(
                    f"Could not analyze any tables in schema '{schema_name or 'default'}'",
                    "data_error"
                )
            
            # Generate ontology
            uri = base_uri or server_config.ontology_base_uri
            generator = OntologyGenerator(base_uri=uri)
            ontology_ttl = generator.generate_from_schema(tables_info)
            
            logger.info(f"Generated ontology for schema '{schema_name or 'default'}': {len(tables_info)} tables")
            return ontology_ttl
            
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
    except Exception as e:
        logger.error(f"Error generating ontology: {e}")
        return create_error_response(
            f"Failed to generate ontology: {str(e)}",
            "internal_error"
        )


def load_ontology_from_file(
    file_path: str
) -> Dict[str, Any]:
    """Load a previously saved or user-edited ontology from the tmp folder.
    
    This tool allows loading ontologies that were:
    - Previously generated and saved by get_analysis_context()  
    - Manually edited by users for enhanced analytical context
    - Created externally and placed in the tmp folder
    
    Args:
        file_path: Path to the ontology file (.ttl format)
                  Can be absolute path or relative to tmp folder
    
    Returns:
        Dictionary containing the loaded ontology content and metadata
        
    Examples:
        # Load a previously saved ontology
        load_ontology_from_file("ontology_public_20240826_143022.ttl")
        
        # Load with full path
        load_ontology_from_file("/path/to/tmp/ontology_custom.ttl")
    """
    try:
        # Get tmp directory
        TMP_DIR = Path(__file__).parent.parent.parent / "tmp"
        
        # Handle relative paths (assume they're in tmp folder)
        if not os.path.isabs(file_path):
            full_path = TMP_DIR / file_path
        else:
            full_path = Path(file_path)
        
        # Check if file exists
        if not full_path.exists():
            return create_error_response(
                f"Ontology file not found: {full_path}",
                "file_not_found",
                f"Available files in tmp folder: {[f.name for f in TMP_DIR.glob('*.ttl')] if TMP_DIR.exists() else 'tmp folder not found'}"
            )
        
        # Check file extension
        if not full_path.suffix.lower() == '.ttl':
            return create_error_response(
                f"Invalid file format. Expected .ttl file, got: {full_path.suffix}",
                "invalid_format",
                "Only Turtle (.ttl) format ontology files are supported"
            )
        
        # Read ontology file
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                ontology_content = f.read()
        except Exception as e:
            return create_error_response(
                f"Failed to read ontology file: {str(e)}",
                "read_error"
            )
        
        # Validate that it's a valid Turtle format (basic check)
        if not ontology_content.strip():
            return create_error_response(
                "Ontology file is empty",
                "empty_file"
            )
        
        # Basic validation - check for common RDF/Turtle patterns
        turtle_patterns = ['@prefix', 'PREFIX', 'rdf:', 'rdfs:', 'owl:', '<', '>']
        if not any(pattern in ontology_content for pattern in turtle_patterns):
            logger.warning(f"File {full_path} may not be a valid Turtle ontology - no RDF patterns found")
        
        # Get file statistics
        file_stats = full_path.stat()
        file_size = file_stats.st_size
        modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Count some basic statistics
        line_count = len(ontology_content.splitlines())
        triple_count = ontology_content.count(' .') + ontology_content.count(' ;\n') + ontology_content.count(' ,\n')
        
        logger.info(f"Successfully loaded ontology from: {full_path} ({file_size} bytes, {line_count} lines)")
        
        return {
            "success": True,
            "ontology": ontology_content,
            "file_info": {
                "file_path": str(full_path),
                "filename": full_path.name,
                "file_size_bytes": file_size,
                "file_size_human": f"{file_size / 1024:.1f} KB" if file_size > 1024 else f"{file_size} B",
                "line_count": line_count,
                "estimated_triple_count": triple_count,
                "last_modified": modified_time
            },
            "usage_hints": [
                "Use this ontology content for analytical context and SQL generation",
                "Extract table.column references for accurate SQL queries",
                "Look for business descriptions to understand data meaning",
                "Check relationship annotations for proper JOIN conditions",
                "Use the ontology as documentation for the database schema"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error loading ontology from file: {e}")
        return create_error_response(
            f"Failed to load ontology: {str(e)}",
            "load_error"
        )