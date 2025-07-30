"""Database connection and schema analysis manager."""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import logging
from dataclasses import dataclass
from contextlib import contextmanager

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


class DatabaseManager:
    """Manages database connections and schema analysis."""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.metadata: Optional[MetaData] = None
        self.connection_info: Dict[str, Any] = {}
        
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
        """Connect to PostgreSQL database."""
        try:
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(
                connection_string,
                pool_timeout=30,
                pool_pre_ping=True,
                echo=False
            )
            self.metadata = MetaData()
            self.connection_info = {
                "type": "postgresql",
                "host": host,
                "port": port,
                "database": database,
                "username": username
            }
            
            # Test connection
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"Connected to PostgreSQL database: {database}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def connect_snowflake(self, account: str, username: str, password: str,
                         warehouse: str, database: str, schema: str = "PUBLIC") -> bool:
        """Connect to Snowflake database."""
        try:
            connection_string = (
                f"snowflake://{username}:{password}@{account}/"
                f"{database}/{schema}?warehouse={warehouse}"
            )
            self.engine = create_engine(
                connection_string,
                pool_timeout=30,
                pool_pre_ping=True,
                echo=False
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
            
            # Test connection
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"Connected to Snowflake database: {database}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            return False
    
    def get_schemas(self) -> List[str]:
        """Get list of available schemas."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        try:
            with self.get_connection() as conn:
                # Use proper parameterized query for schema filtering
                if self.connection_info.get("type") == "snowflake":
                    query = text("""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name NOT IN ('INFORMATION_SCHEMA', 'SNOWFLAKE', 'SNOWFLAKE_SAMPLE_DATA')
                        ORDER BY schema_name
                    """)
                else:
                    query = text("""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
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
                
                # Get row count using safe parameterized query
                row_count = None
                try:
                    if schema_name:
                        # Use SQLAlchemy's quoted_name for safe identifier quoting
                        count_query = text("SELECT COUNT(*) FROM :schema_table")
                        # For proper schema.table handling, we need to construct this differently
                        if self.connection_info.get("type") == "snowflake":
                            full_table_name = f'"{schema_name}"."{table_name}"'
                        else:
                            full_table_name = f'"{schema_name}"."{table_name}"'
                        
                        # Use a safer approach with string formatting but validate inputs first
                        if self._validate_identifier(schema_name) and self._validate_identifier(table_name):
                            count_query = text(f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"')
                            result = conn.execute(count_query)
                            row_count = result.scalar()
                    else:
                        if self._validate_identifier(table_name):
                            count_query = text(f'SELECT COUNT(*) FROM "{table_name}"')
                            result = conn.execute(count_query)
                            row_count = result.scalar()
                except SQLAlchemyError as e:
                    logger.warning(f"Could not get row count for {table_name}: {e}")
                    # Row count might fail for various reasons, continue without it
                    pass
                
                return TableInfo(
                    name=table_name,
                    schema=schema_name or "public",
                    columns=columns,
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys,
                    comment=None,  # Would need separate query for table comments
                    row_count=row_count
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")
            return None
    
    def _validate_identifier(self, identifier: str) -> bool:
        """Validate database identifier to prevent injection."""
        if not identifier:
            return False
        # Allow alphanumeric, underscore, and hyphens only
        return identifier.replace('_', '').replace('-', '').isalnum()
    
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
                         limit: int = 10) -> List[Dict[str, Any]]:
        """Sample data from a table for analysis."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        # Validate inputs
        if not self._validate_identifier(table_name):
            logger.error(f"Invalid table name: {table_name}")
            return []
        
        if schema_name and not self._validate_identifier(schema_name):
            logger.error(f"Invalid schema name: {schema_name}")
            return []
        
        if not isinstance(limit, int) or limit <= 0 or limit > 1000:
            limit = 10
        
        try:
            with self.get_connection() as conn:
                if schema_name:
                    # Construct safe query with validated identifiers
                    query = text(f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT :limit')
                else:
                    query = text(f'SELECT * FROM "{table_name}" LIMIT :limit')
                
                result = conn.execute(query, {"limit": limit})
                columns = list(result.keys())
                
                rows = []
                for row in result.fetchall():
                    row_dict = {}
                    for i, value in enumerate(row):
                        # Convert non-serializable types to strings
                        if value is not None:
                            if hasattr(value, 'isoformat'):  # datetime objects
                                row_dict[columns[i]] = value.isoformat()
                            else:
                                row_dict[columns[i]] = value
                        else:
                            row_dict[columns[i]] = None
                    rows.append(row_dict)
                
                return rows
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to sample data from {table_name}: {e}")
            return []
    
    def disconnect(self):
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.metadata = None
            self.connection_info = {}
            logger.info("Database connection closed")
