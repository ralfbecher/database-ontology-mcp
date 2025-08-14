"""Database connection and schema analysis manager."""

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError
from sqlalchemy.pool import QueuePool

from .constants import (
    CONNECTION_TIMEOUT, 
    QUERY_TIMEOUT, 
    IDENTIFIER_PATTERN,
    POSTGRES_SYSTEM_SCHEMAS,
    SNOWFLAKE_SYSTEM_SCHEMAS,
    MIN_SAMPLE_LIMIT,
    MAX_SAMPLE_LIMIT,
    DEFAULT_SAMPLE_LIMIT
)

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    foreign_key_table: Optional[str] = None
    foreign_key_column: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    schema: str
    columns: List[ColumnInfo]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    comment: Optional[str] = None
    row_count: Optional[int] = None
    sample_data: Optional[List[Dict[str, Any]]] = None


class DatabaseManager:
    """Manages database connections and schema analysis."""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.metadata: Optional[MetaData] = None
        self.connection_info: Dict[str, Any] = {}
        self._connection_pool_size = 5
        self._max_overflow = 10
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
        
    def connect_postgresql(self, host: str, port: int, database: str, 
                          username: str, password: str) -> bool:
        """Connect to PostgreSQL database with enhanced security and reliability."""
        try:
            # Validate inputs
            if not all([host, port, database, username]):
                logger.error("Missing required PostgreSQL connection parameters")
                return False
            
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self._connection_pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=CONNECTION_TIMEOUT,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                echo=False,
                connect_args={
                    "connect_timeout": CONNECTION_TIMEOUT,
                    "application_name": "database-ontology-mcp"
                }
            )
            self.metadata = MetaData()
            self.connection_info = {
                "type": "postgresql",
                "host": host,
                "port": port,
                "database": database,
                "username": username
            }
            
            # Test connection with timeout
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info(f"Connected to PostgreSQL database: {database} at {host}:{port}")
            return True
            
        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(f"Failed to connect to PostgreSQL {host}:{port}/{database}: {type(e).__name__}: {e}")
            self.engine = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to PostgreSQL: {type(e).__name__}: {e}")
            self.engine = None
            return False
    
    def connect_snowflake(self, account: str, username: str, password: str,
                         warehouse: str, database: str, schema: str = "PUBLIC") -> bool:
        """Connect to Snowflake database with enhanced security and reliability."""
        try:
            # Validate inputs
            if not all([account, username, warehouse, database]):
                logger.error("Missing required Snowflake connection parameters")
                return False
            
            connection_string = (
                f"snowflake://{username}:{password}@{account}/"
                f"{database}/{schema}?warehouse={warehouse}"
            )
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self._connection_pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=CONNECTION_TIMEOUT,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
                connect_args={
                    "application": "database-ontology-mcp",
                    "network_timeout": CONNECTION_TIMEOUT
                }
            )
            self.metadata = MetaData()
            self.connection_info = {
                "type": "snowflake",
                "account": account,
                "username": username,
                "warehouse": warehouse,
                "database": database,
                "schema": schema
            }
            
            # Test connection with timeout
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info(f"Connected to Snowflake database: {database} (warehouse: {warehouse})")
            return True
            
        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(f"Failed to connect to Snowflake {account}/{database}: {type(e).__name__}: {e}")
            self.engine = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Snowflake: {type(e).__name__}: {e}")
            self.engine = None
            return False
    
    def get_schemas(self) -> List[str]:
        """Get list of available schemas."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        try:
            with self.get_connection() as conn:
                # Use proper parameterized query for schema filtering
                if self.connection_info.get("type") == "snowflake":
                    excluded_schemas = "', '".join(SNOWFLAKE_SYSTEM_SCHEMAS)
                    query = text(f"""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name NOT IN ('{excluded_schemas}')
                        ORDER BY schema_name
                    """)
                else:
                    excluded_schemas = "', '".join(POSTGRES_SYSTEM_SCHEMAS)
                    query = text(f"""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name NOT IN ('{excluded_schemas}')
                        ORDER BY schema_name
                    """)
                result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get schemas: {e}")
            return []
    
    def get_tables(self, schema_name: Optional[str] = None) -> List[str]:
        """Get list of tables in a schema."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        try:
            with self.get_connection() as conn:
                if schema_name:
                    query = text("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = :schema_name 
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """)
                    result = conn.execute(query, {"schema_name": schema_name})
                else:
                    query = text("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """)
                    result = conn.execute(query)
                
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get tables: {e}")
            return []
    
    def analyze_table(self, table_name: str, schema_name: Optional[str] = None) -> Optional[TableInfo]:
        """Analyze a specific table and return detailed information."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        try:
            with self.get_connection() as conn:
                # Use SQLAlchemy inspector for safer metadata reflection
                inspector = inspect(self.engine)
                
                # Get table metadata using inspector
                if schema_name:
                    if not inspector.has_table(table_name, schema=schema_name):
                        logger.error(f"Table {schema_name}.{table_name} not found")
                        return None
                    table_columns = inspector.get_columns(table_name, schema=schema_name)
                    table_pk = inspector.get_pk_constraint(table_name, schema=schema_name)
                    table_fks = inspector.get_foreign_keys(table_name, schema=schema_name)
                else:
                    if not inspector.has_table(table_name):
                        logger.error(f"Table {table_name} not found")
                        return None
                    table_columns = inspector.get_columns(table_name)
                    table_pk = inspector.get_pk_constraint(table_name)
                    table_fks = inspector.get_foreign_keys(table_name)
                
                # Process column information
                columns = []
                primary_keys = table_pk.get('constrained_columns', []) if table_pk else []
                foreign_keys = []
                
                for col_info in table_columns:
                    column_name = col_info['name']
                    is_pk = column_name in primary_keys
                    
                    # Check for foreign keys
                    fk_table = None
                    fk_column = None
                    is_fk = False
                    
                    for fk in table_fks:
                        if column_name in fk['constrained_columns']:
                            is_fk = True
                            fk_idx = fk['constrained_columns'].index(column_name)
                            fk_table = fk['referred_table']
                            fk_column = fk['referred_columns'][fk_idx]
                            foreign_keys.append({
                                "column": column_name,
                                "referenced_table": fk_table,
                                "referenced_column": fk_column
                            })
                            break
                    
                    column_info = ColumnInfo(
                        name=column_name,
                        data_type=str(col_info['type']),
                        is_nullable=col_info['nullable'],
                        is_primary_key=is_pk,
                        is_foreign_key=is_fk,
                        foreign_key_table=fk_table,
                        foreign_key_column=fk_column,
                        comment=col_info.get('comment')
                    )
                    columns.append(column_info)
                
                # Get row count and sample data
                row_count = None
                sample_data = None
                
                try:
                    # Validate identifiers once
                    if not self._validate_identifier(table_name) or (schema_name and not self._validate_identifier(schema_name)):
                        logger.warning(f"Invalid identifier format: {schema_name}.{table_name}")
                    else:
                        # Construct table name once
                        if schema_name:
                            full_table_name = f'"{schema_name}"."{table_name}"'
                        else:
                            full_table_name = f'"{table_name}"'
                        
                        # Get row count
                        count_query = text(f'SELECT COUNT(*) FROM {full_table_name}')
                        result = conn.execute(count_query)
                        row_count = result.scalar()
                        
                        # Get 10 random sample rows
                        if row_count and row_count > 0:
                            if self.connection_info.get("type") == "snowflake":
                                # Snowflake uses SAMPLE for random sampling
                                sample_query = text(f'SELECT * FROM {full_table_name} SAMPLE (10 ROWS) LIMIT 10')
                            else:
                                # PostgreSQL uses ORDER BY RANDOM() for random sampling
                                sample_query = text(f'SELECT * FROM {full_table_name} ORDER BY RANDOM() LIMIT 10')
                            
                            sample_result = conn.execute(sample_query)
                            sample_columns = list(sample_result.keys())
                            
                            sample_rows = []
                            for row in sample_result.fetchall():
                                row_dict = {}
                                for i, value in enumerate(row):
                                    column_name = sample_columns[i]
                                    if value is not None:
                                        if hasattr(value, 'isoformat'):  # datetime objects
                                            row_dict[column_name] = value.isoformat()
                                        elif isinstance(value, (bytes, bytearray)):
                                            row_dict[column_name] = value.hex()
                                        elif hasattr(value, '__dict__'):  # Complex objects
                                            row_dict[column_name] = str(value)
                                        else:
                                            row_dict[column_name] = value
                                    else:
                                        row_dict[column_name] = None
                                sample_rows.append(row_dict)
                            
                            sample_data = sample_rows
                        
                except SQLAlchemyError as e:
                    logger.warning(f"Could not get row count or sample data for {table_name}: {e}")
                    # Continue without row count and sample data
                
                return TableInfo(
                    name=table_name,
                    schema=schema_name or "public",
                    columns=columns,
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys,
                    comment=None,  # Would need separate query for table comments
                    row_count=row_count,
                    sample_data=sample_data
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")
            return None
    
    def _validate_identifier(self, identifier: str) -> bool:
        """Validate database identifier to prevent injection attacks."""
        if not identifier or len(identifier) > 63:  # PostgreSQL limit
            return False
        
        # Use regex pattern for strict validation
        return bool(re.match(IDENTIFIER_PATTERN, identifier))
    
    def get_table_relationships(self, schema_name: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """Get relationships between tables in a schema."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        relationships = {}
        tables = self.get_tables(schema_name)
        
        for table_name in tables:
            table_info = self.analyze_table(table_name, schema_name)
            if table_info and table_info.foreign_keys:
                relationships[table_name] = table_info.foreign_keys
        
        return relationships
    
    def sample_table_data(self, table_name: str, schema_name: Optional[str] = None, 
                         limit: int = DEFAULT_SAMPLE_LIMIT) -> List[Dict[str, Any]]:
        """Sample data from a table for analysis with enhanced validation."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        # Validate inputs
        if not self._validate_identifier(table_name):
            logger.error(f"Invalid table name format: {table_name}")
            raise ValueError(f"Invalid table name format: {table_name}")
        
        if schema_name and not self._validate_identifier(schema_name):
            logger.error(f"Invalid schema name format: {schema_name}")
            raise ValueError(f"Invalid schema name format: {schema_name}")
        
        # Validate and normalize limit
        if not isinstance(limit, int) or limit < MIN_SAMPLE_LIMIT:
            limit = DEFAULT_SAMPLE_LIMIT
        elif limit > MAX_SAMPLE_LIMIT:
            limit = MAX_SAMPLE_LIMIT
            logger.warning(f"Sample limit capped at {MAX_SAMPLE_LIMIT}")
        
        try:
            with self.get_connection() as conn:
                # Use SQLAlchemy's text with bound parameters for safety
                if schema_name:
                    # Double-quote identifiers to handle case sensitivity
                    full_table_name = f'"{schema_name}"."{table_name}"'
                else:
                    full_table_name = f'"{table_name}"'
                
                query = text(f'SELECT * FROM {full_table_name} LIMIT :limit')
                result = conn.execute(query, {"limit": limit})
                columns = list(result.keys())
                
                rows = []
                for row in result.fetchall():
                    row_dict = {}
                    for i, value in enumerate(row):
                        # Convert non-serializable types to strings
                        column_name = columns[i]
                        if value is not None:
                            if hasattr(value, 'isoformat'):  # datetime objects
                                row_dict[column_name] = value.isoformat()
                            elif isinstance(value, (bytes, bytearray)):
                                # Convert binary data to hex string
                                row_dict[column_name] = value.hex()
                            elif hasattr(value, '__dict__'):  # Complex objects
                                row_dict[column_name] = str(value)
                            else:
                                row_dict[column_name] = value
                        else:
                            row_dict[column_name] = None
                    rows.append(row_dict)
                
                return rows
                
        except (SQLAlchemyError, ValueError) as e:
            logger.error(f"Failed to sample data from {table_name}: {type(e).__name__}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error sampling {table_name}: {type(e).__name__}: {e}")
            return []
    
    def disconnect(self):
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.metadata = None
            self.connection_info = {}
            logger.info("Database connection closed")
