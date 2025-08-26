"""Schema analysis and data sampling tools."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from ..config import config_manager
from ..database_manager import TableInfo, ColumnInfo
from ..ontology_generator import OntologyGenerator
from ..shared import get_db_manager, create_error_response
from ..utils import sanitize_for_logging

logger = logging.getLogger(__name__)


@contextmanager
def error_handler(operation_name: str):
    """Context manager for consistent error handling."""
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation_name}: {type(e).__name__}: {e}")
        raise


def list_schemas() -> Dict[str, Any]:
    """Get a list of available schemas from the connected database.
    
    Returns:
        List of schema names or error response
    """
    with error_handler("list_schemas") as handler:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error", 
                "Use connect_database tool first"
            )
        
        try:
            schemas = db_manager.get_schemas()
            logger.debug(f"Retrieved {len(schemas)} schemas")
            return {
                "success": True,
                "schemas": schemas,
                "count": len(schemas)
            }
        except Exception as e:
            logger.error(f"Failed to list schemas: {e}")
            return create_error_response(
                "Failed to retrieve schema list",
                "database_error",
                str(e)
            )


def get_analysis_context(
    schema_name: Optional[str] = None,
    include_ontology: bool = True
) -> Dict[str, Any]:
    """🌟 MAIN TOOL: Get comprehensive analysis context for data exploration and SQL generation.
    
    This is the primary tool for database analysis. It provides everything needed in one call:
    - Complete schema structure (tables, columns, relationships)  
    - Automatic ontology generation with SQL references
    - Ready-to-use JOIN conditions and column references
    - Relationship warnings for safe aggregations
    - SQL generation hints and best practices
    
    Args:
        schema_name: Name of the schema to analyze (optional)
        include_ontology: Whether to generate ontology (default: True, recommended)
    
    Returns:
        Dictionary containing complete analysis context with schema and ontology data
    """
    try:
        db_manager = get_db_manager()
        server_config = config_manager.get_server_config()
        
        # Check connection
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        logger.info(f"Generating complete analysis context for schema: {schema_name or 'default'}")
        
        # Get schema analysis (inline implementation)
        try:
            tables = db_manager.get_tables(schema_name)
            logger.debug(f"Found {len(tables)} tables in schema '{schema_name or 'default'}'")
            
            # Sequential table analysis 
            all_table_info = []
            for table_name in tables:
                try:
                    table_info = db_manager.analyze_table(table_name, schema_name)
                    if table_info:
                        # Convert dataclass to dict for JSON serialization
                        table_dict = {
                            "name": table_info.name,
                            "schema": table_info.schema,
                            "columns": [
                                {
                                    "name": col.name,
                                    "data_type": col.data_type,
                                    "is_nullable": col.is_nullable,
                                    "is_primary_key": col.is_primary_key,
                                    "is_foreign_key": col.is_foreign_key,
                                    "foreign_key_table": col.foreign_key_table,
                                    "foreign_key_column": col.foreign_key_column,
                                    "comment": col.comment
                                } for col in table_info.columns
                            ],
                            "primary_keys": table_info.primary_keys,
                            "foreign_keys": table_info.foreign_keys,
                            "comment": table_info.comment,
                            "row_count": table_info.row_count,
                            "sample_data": table_info.sample_data
                        }
                        all_table_info.append(table_dict)
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table_name}: {e}")
            
            schema_data = {
                "schema": schema_name or "default",
                "table_count": len(all_table_info),
                "tables": all_table_info
            }
            
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        # Get relationships (inline implementation)
        relationships = {}
        table_names = [table['name'] for table in all_table_info]
        
        for table_name in table_names:
            table_data = next((t for t in all_table_info if t['name'] == table_name), None)
            if table_data and table_data.get('foreign_keys'):
                relationships[table_name] = table_data['foreign_keys']
            
        result = {
            "schema_analysis": schema_data,
            "relationships": relationships,
            "ontology": None,
            "sql_hints": {
                "workflow": [
                    "1. Review the schema_analysis to understand table structure",
                    "2. Use the ontology for business context and SQL references",
                    "3. Check relationships for potential fan-traps before JOINs",
                    "4. Validate SQL syntax before execution",
                    "5. Execute queries with appropriate limits"
                ]
            }
        }
        
        # Generate ontology if requested (inline implementation)
        if include_ontology:
            try:
                # Convert schema data to TableInfo objects for ontology generation
                tables_info = []
                for table_dict in all_table_info:
                    columns = []
                    for col_dict in table_dict['columns']:
                        col_info = ColumnInfo(
                            name=col_dict['name'],
                            data_type=col_dict['data_type'],
                            is_nullable=col_dict['is_nullable'],
                            is_primary_key=col_dict['is_primary_key'],
                            is_foreign_key=col_dict['is_foreign_key'],
                            foreign_key_table=col_dict['foreign_key_table'],
                            foreign_key_column=col_dict['foreign_key_column'],
                            comment=col_dict['comment']
                        )
                        columns.append(col_info)
                    
                    table_info = TableInfo(
                        name=table_dict['name'],
                        schema=table_dict['schema'],
                        columns=columns,
                        primary_keys=table_dict['primary_keys'],
                        foreign_keys=table_dict['foreign_keys'],
                        comment=table_dict['comment'],
                        row_count=table_dict['row_count'],
                        sample_data=table_dict['sample_data']
                    )
                    tables_info.append(table_info)
                
                # Generate ontology
                uri = server_config.ontology_base_uri
                generator = OntologyGenerator(base_uri=uri)
                ontology_ttl = generator.generate_from_schema(tables_info)
                
                if ontology_ttl:
                    result["ontology"] = ontology_ttl
                    
                    # Save ontology to tmp folder for user editing
                    try:
                        from pathlib import Path
                        TMP_DIR = Path(__file__).parent.parent.parent / "tmp"
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        schema_safe = (schema_name or "default").replace(" ", "_").replace(".", "_")
                        ontology_filename = f"ontology_{schema_safe}_{timestamp}.ttl"
                        ontology_file_path = TMP_DIR / ontology_filename
                        
                        with open(ontology_file_path, 'w', encoding='utf-8') as f:
                            f.write(ontology_ttl)
                        
                        result["ontology_file_path"] = str(ontology_file_path)
                        logger.info(f"Saved ontology to: {ontology_file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save ontology to file: {e}")
                    
                    result["sql_hints"]["ontology_benefits"] = [
                        "Contains ready-to-use SQL column references (e.g., customers.customer_id)",
                        "Includes complete JOIN conditions for relationships", 
                        "Provides business descriptions for understanding data meaning",
                        "Shows data types, constraints, and row counts",
                        "Acts as both documentation and SQL generation reference",
                        "Saved to tmp folder for manual editing if needed"
                    ]
                else:
                    logger.warning("Failed to generate ontology for analysis context")
            except Exception as e:
                logger.warning(f"Could not generate ontology for analysis context: {e}")
        
        # Add relationship warnings for analysis
        fan_trap_warnings = []
        for table, fks in relationships.items():
            if len(fks) > 1:
                referenced_tables = [fk['referenced_table'] for fk in fks]
                fan_trap_warnings.append({
                    "table": table,
                    "warning": f"Table {table} connects to multiple tables - potential fan-trap risk",
                    "referenced_tables": referenced_tables,
                    "recommendation": "Use separate CTEs or UNION approach for multi-fact aggregations"
                })
        
        if fan_trap_warnings:
            result["sql_hints"]["fan_trap_warnings"] = fan_trap_warnings
            
        logger.info(f"Generated analysis context: {len(schema_data.get('tables', []))} tables, "
                   f"ontology: {result['ontology'] is not None}")
                   
        return result
        
    except Exception as e:
        logger.error(f"Error generating analysis context: {e}")
        return create_error_response(
            f"Failed to generate analysis context: {str(e)}",
            "internal_error"
        )


def sample_table_data(
    table_name: str,
    schema_name: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """Sample data from a specific table for exploration and analysis.
    
    Args:
        table_name: Name of the table to sample
        schema_name: Name of the schema containing the table (optional)
        limit: Maximum number of rows to return (default: 10, max: 1000)
    
    Returns:
        List of dictionaries representing sample rows or error response
    """
    with error_handler("sample_table_data") as handler:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        try:
            sample_data = db_manager.sample_table_data(table_name, schema_name, limit)
            logger.info(f"Retrieved {len(sample_data)} sample rows from {table_name}")
            return {
                "success": True,
                "table_name": table_name,
                "schema_name": schema_name,
                "sample_data": sample_data,
                "row_count": len(sample_data),
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Failed to sample table data: {e}")
            return create_error_response(
                f"Failed to sample data from table '{table_name}'",
                "database_error",
                str(e)
            )