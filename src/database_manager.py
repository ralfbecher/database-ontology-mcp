"""Database connection and schema analysis manager."""

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union

from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError, ProgrammingError
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import sqltypes

from .constants import (
    CONNECTION_TIMEOUT, 
    QUERY_TIMEOUT, 
    IDENTIFIER_PATTERN,
    POSTGRES_SYSTEM_SCHEMAS,
    SNOWFLAKE_SYSTEM_SCHEMAS,
    DREMIO_SYSTEM_SCHEMAS,
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
    """Manages database connections and schema analysis with enhanced reliability."""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.metadata: Optional[MetaData] = None
        self.connection_info: Dict[str, Any] = {}
        self._connection_pool_size = 5
        self._max_overflow = 10
        self._last_connection_params: Optional[Dict[str, Any]] = None
        
    def _test_connection(self) -> bool:
        """Test if the current connection is healthy."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False
    
    def _ensure_connection(self):
        """Ensure we have a healthy database connection, reconnecting if necessary."""
        logger.debug(f"_ensure_connection: engine exists: {self.engine is not None}")
        logger.debug(f"_ensure_connection: last_params available: {self._last_connection_params is not None}")
        
        if not self.engine:
            if self._last_connection_params:
                logger.info("No engine found, reconnecting to database using stored parameters")
                self._reconnect()
            else:
                raise RuntimeError("No database connection established and no connection parameters available")
        elif not self._test_connection():
            if self._last_connection_params:
                logger.info("Connection health check failed, reconnecting to database")
                self._reconnect()
            else:
                logger.error("Connection unhealthy but no reconnection parameters available")
                raise RuntimeError("Database connection is unhealthy and cannot be restored")
                
        logger.debug(f"_ensure_connection: final engine state: {self.engine is not None}")
    
    def _reconnect(self):
        """Reconnect to the database using stored parameters."""
        if not self._last_connection_params:
            raise RuntimeError("No connection parameters stored for reconnection")
        
        params = self._last_connection_params
        if params["type"] == "postgresql":
            success = self.connect_postgresql(
                params["host"], params["port"], params["database"],
                params["username"], params["password"]
            )
        elif params["type"] == "snowflake":
            success = self.connect_snowflake(
                params["account"], params["username"], params["password"],
                params["warehouse"], params["database"], params.get("schema", "PUBLIC")
            )
        else:  # dremio
            success = self.connect_dremio(
                params["host"], params["port"], params["username"], params["password"],
                params.get("ssl", True)
            )
        
        if not success:
            raise RuntimeError(f"Failed to reconnect to {params['type']} database")
        
        logger.info(f"Successfully reconnected to {params['type']} database")
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with auto-reconnection."""
        # Basic connection check first
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        # Try connection with retry logic
        max_retries = 2
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                conn = self.engine.connect()
                try:
                    yield conn
                finally:
                    conn.close()
                return  # Success, exit method
            except (OperationalError, DatabaseError) as e:
                last_exception = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:  # Not the last attempt
                    if self._last_connection_params:
                        logger.info("Attempting reconnection...")
                        try:
                            self._reconnect()
                        except Exception as reconnect_error:
                            logger.error(f"Reconnection failed: {reconnect_error}")
                            # Continue to next attempt or fail
                    else:
                        logger.error("No connection parameters available for reconnection")
                        break
                        
        # If we get here, all attempts failed
        logger.error(f"All connection attempts failed. Last error: {last_exception}")
        raise RuntimeError(f"Database connection failed after {max_retries} attempts: {last_exception}")
        
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
            
            # Store connection parameters for reconnection
            self._last_connection_params = {
                "type": "postgresql",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password
            }
            
            # Test connection with timeout
            with self.engine.connect() as conn:
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
                         warehouse: str, database: str, schema: str = "PUBLIC", role: str = "PUBLIC") -> bool:
        """Connect to Snowflake database with enhanced security and reliability."""
        try:
            # Validate inputs
            if not all([account, username, warehouse, database]):
                logger.error("Missing required Snowflake connection parameters")
                return False
            
            # URL-encode password to handle special characters
            from urllib.parse import quote_plus
            encoded_password = quote_plus(password) if password else ""
            
            connection_string = (
                f"snowflake://{username}:{encoded_password}@{account}/"
                f"{database}/{schema}?warehouse={warehouse}&role={role}"
            )
            
            logger.info(f"Connecting to Snowflake with account: {account}, database: {database}, warehouse: {warehouse}, schema: {schema}")
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
                "schema": schema,
                "role": role
            }
            
            # Store connection parameters for reconnection
            self._last_connection_params = {
                "type": "snowflake",
                "account": account,
                "username": username,
                "password": password,
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
                "role": role
            }
            
            # Test connection with timeout
            with self.engine.connect() as conn:
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
    
    def connect_dremio(self, host: str, port: int, username: str, password: str, ssl: bool = True) -> bool:
        """Connect to Dremio database with enhanced security and reliability."""
        try:
            # Validate inputs
            if not all([host, port, username]):
                logger.error("Missing required Dremio connection parameters")
                return False
            
            # Dremio supports PostgreSQL wire protocol
            from urllib.parse import quote_plus
            encoded_password = quote_plus(password) if password else ""
            
            # Dremio uses PostgreSQL protocol but connects to a virtual database
            connection_string = (
                f"postgresql://{username}:{encoded_password}@{host}:{port}/dremio"
                f"?sslmode={'require' if ssl else 'disable'}"
                f"&application_name=database-ontology-mcp"
            )
            
            logger.info(f"Connecting to Dremio at {host}:{port} (SSL: {ssl})")
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
                    "timeout": CONNECTION_TIMEOUT,
                    "application_name": "database-ontology-mcp"
                }
            )
            self.metadata = MetaData()
            self.connection_info = {
                "type": "dremio",
                "host": host,
                "port": port,
                "username": username,
                "ssl": ssl
            }
            
            # Store connection parameters for reconnection
            self._last_connection_params = {
                "type": "dremio",
                "host": host,
                "port": port,
                "username": username,
                "password": password,
                "ssl": ssl
            }
            
            # Test connection with timeout
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info(f"Connected to Dremio database at {host}:{port}")
            return True
            
        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(f"Failed to connect to Dremio {host}:{port}: {type(e).__name__}: {e}")
            self.engine = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Dremio: {type(e).__name__}: {e}")
            self.engine = None
            return False
    
    def get_schemas(self) -> List[str]:
        """Get list of available schemas."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        try:
            with self.get_connection() as conn:
                # Use proper parameterized query for schema filtering
                db_type = self.connection_info.get("type")
                if db_type == "snowflake":
                    excluded_schemas = "', '".join(SNOWFLAKE_SYSTEM_SCHEMAS)
                elif db_type == "dremio":
                    excluded_schemas = "', '".join(DREMIO_SYSTEM_SCHEMAS)
                else:  # postgresql
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
        logger.debug(f"get_tables: Starting, engine exists: {self.has_engine()}")
        
        # Ensure connection before proceeding
        try:
            self._ensure_connection()
        except RuntimeError as e:
            logger.error(f"get_tables: Connection check failed: {e}")
            raise
        
        logger.debug(f"get_tables: After ensure_connection, engine exists: {self.has_engine()}")
        
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
        logger.debug(f"analyze_table: Starting analysis of {table_name}, engine exists: {self.has_engine()}")
        
        # Ensure connection before proceeding
        try:
            self._ensure_connection()
        except RuntimeError as e:
            logger.error(f"analyze_table: Connection check failed: {e}")
            raise
        
        logger.debug(f"analyze_table: After ensure_connection, engine exists: {self.has_engine()}")
        
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
                            db_type = self.connection_info.get("type")
                            if db_type == "snowflake":
                                # Snowflake uses SAMPLE for random sampling
                                sample_query = text(f'SELECT * FROM {full_table_name} SAMPLE (10 ROWS) LIMIT 10')
                            elif db_type == "dremio":
                                # Dremio doesn't support TABLESAMPLE or RANDOM(), use simple LIMIT
                                sample_query = text(f'SELECT * FROM {full_table_name} LIMIT 10')
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
    
    def _strip_leading_sql_comments(self, sql_query: str) -> str:
        """Strip leading SQL comments to find the actual SQL statement.
        
        Handles both -- (line comments) and /* */ (block comments) at the beginning of queries.
        
        Args:
            sql_query: SQL query that may contain leading comments
            
        Returns:
            SQL query with leading comments removed
        """
        import re
        
        lines = sql_query.split('\n')
        result_lines = []
        in_block_comment = False
        
        for line in lines:
            original_line = line
            line = line.strip()
            
            # Handle block comment continuation
            if in_block_comment:
                if '*/' in line:
                    # End of block comment found
                    after_comment = line.split('*/', 1)[1].strip()
                    in_block_comment = False
                    if after_comment:  # If there's SQL after the comment
                        result_lines.append(after_comment)
                continue
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle line comments
            if line.startswith('--'):
                continue
                
            # Handle block comments
            if line.startswith('/*'):
                if '*/' in line:
                    # Single-line block comment
                    after_comment = line.split('*/', 1)[1].strip()
                    if after_comment:  # If there's SQL after the comment
                        result_lines.append(after_comment)
                        break
                else:
                    # Multi-line block comment starts
                    in_block_comment = True
                continue
            
            # This line contains actual SQL - add it and all remaining lines
            result_lines.append(original_line)
            # Add all remaining lines without further comment processing
            remaining_index = lines.index(original_line) + 1
            if remaining_index < len(lines):
                result_lines.extend(lines[remaining_index:])
            break
        
        return '\n'.join(result_lines).strip()
    
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
    
    def validate_sql_syntax(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query syntax using database-level validation.
        
        Uses the database's own SQL parser via prepared statements to provide
        accurate syntax validation and meaningful error messages for LLM correction.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Dictionary with validation results including database-specific errors
        """
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        validation_result = {
            "is_valid": False,
            "error": None,
            "error_type": None,
            "database_error": None,
            "query_type": None,
            "affected_tables": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # Step 1: Basic security checks
            query_stripped = sql_query.strip()
            if not query_stripped:
                validation_result["error"] = "Empty query"
                validation_result["error_type"] = "empty_query"
                return validation_result
            
            # Step 2: Remove leading SQL comments to find actual SQL statement
            query_without_comments = self._strip_leading_sql_comments(query_stripped)
            query_upper = query_without_comments.upper()
            
            # Check for multiple statements (security risk) - use original query for this check
            if ';' in query_stripped[:-1]:  # Allow trailing semicolon
                validation_result["error"] = "Multiple SQL statements not allowed for security"
                validation_result["error_type"] = "security_error"
                validation_result["suggestions"].append("Split multiple statements into separate requests")
                return validation_result
            
            # Determine and validate query type using comment-stripped query
            if query_upper.startswith('SELECT'):
                validation_result["query_type"] = "SELECT"
            elif query_upper.startswith('WITH'):
                validation_result["query_type"] = "CTE_SELECT"
                if 'SELECT' not in query_upper:
                    validation_result["warnings"].append("CTE should end with SELECT statement")
            elif query_upper.startswith(('EXPLAIN', 'DESCRIBE', 'DESC', 'SHOW')):
                validation_result["query_type"] = "METADATA"
            else:
                # Check for potentially dangerous operations
                dangerous_ops = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE', 'MERGE']
                detected_ops = [op for op in dangerous_ops if query_upper.startswith(op)]
                if detected_ops:
                    validation_result["error"] = f"Destructive operations not allowed: {', '.join(detected_ops)}"
                    validation_result["error_type"] = "forbidden_operation"
                    validation_result["suggestions"].append("Use SELECT queries for data retrieval only")
                    return validation_result
                else:
                    validation_result["error"] = "Only SELECT, CTE, and metadata queries are allowed"
                    validation_result["error_type"] = "query_type_error"
                    validation_result["suggestions"].append("Start your query with SELECT, WITH, EXPLAIN, or SHOW")
                    return validation_result
            
            # Step 2: Database-level syntax validation
            with self.get_connection() as conn:
                try:
                    if self.connection_info.get("type") == "postgresql":
                        # PostgreSQL: Use PREPARE to validate syntax without execution
                        prepare_name = "syntax_check_stmt"
                        prepare_sql = f"PREPARE {prepare_name} AS {query_stripped}"
                        deallocate_sql = f"DEALLOCATE {prepare_name}"
                        
                        try:
                            # Prepare the statement (validates syntax)
                            conn.execute(text(prepare_sql))
                            # Clean up the prepared statement
                            conn.execute(text(deallocate_sql))
                            validation_result["is_valid"] = True
                        except Exception as prepare_error:
                            # Extract meaningful error from PostgreSQL
                            error_msg = str(prepare_error)
                            validation_result["database_error"] = error_msg
                            validation_result["error"] = f"PostgreSQL syntax error: {error_msg}"
                            validation_result["error_type"] = "syntax_error"
                            
                            # Add suggestions based on common errors
                            if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
                                validation_result["suggestions"].append("Check table/column names - they may not exist or may need proper schema qualification")
                            elif "syntax error" in error_msg.lower():
                                validation_result["suggestions"].append("Review SQL syntax - check for missing commas, parentheses, or keywords")
                            elif "permission denied" in error_msg.lower():
                                validation_result["suggestions"].append("Insufficient permissions to access the specified tables")
                            
                    elif self.connection_info.get("type") == "snowflake":
                        # Snowflake: Use EXPLAIN to validate syntax
                        explain_sql = f"EXPLAIN {query_stripped}"
                        try:
                            result = conn.execute(text(explain_sql))
                            result.fetchall()  # Consume the explain plan
                            validation_result["is_valid"] = True
                        except Exception as explain_error:
                            # Extract meaningful error from Snowflake
                            error_msg = str(explain_error)
                            validation_result["database_error"] = error_msg
                            validation_result["error"] = f"Snowflake syntax error: {error_msg}"
                            validation_result["error_type"] = "syntax_error"
                            
                            # Add suggestions based on common Snowflake errors
                            if "does not exist" in error_msg.lower():
                                validation_result["suggestions"].append("Object not found - check table/schema names and ensure proper qualification (DATABASE.SCHEMA.TABLE)")
                            elif "sql compilation error" in error_msg.lower():
                                validation_result["suggestions"].append("SQL compilation failed - review syntax and object references")
                            elif "invalid identifier" in error_msg.lower():
                                validation_result["suggestions"].append("Invalid identifier - check column names and use double quotes for case-sensitive names")
                    
                    elif self.connection_info.get("type") == "dremio":
                        # Dremio: Use EXPLAIN to validate syntax
                        explain_sql = f"EXPLAIN PLAN FOR {query_stripped}"
                        try:
                            result = conn.execute(text(explain_sql))
                            result.fetchall()  # Consume the explain plan
                            validation_result["is_valid"] = True
                        except Exception as explain_error:
                            # Extract meaningful error from Dremio
                            error_msg = str(explain_error)
                            validation_result["database_error"] = error_msg
                            validation_result["error"] = f"Dremio syntax error: {error_msg}"
                            validation_result["error_type"] = "syntax_error"
                            
                            # Add suggestions based on common Dremio errors
                            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                                validation_result["suggestions"].append("Object not found - check table/view names and ensure proper qualification")
                            elif "syntax error" in error_msg.lower():
                                validation_result["suggestions"].append("SQL syntax error - review query structure and keywords")
                            elif "validation error" in error_msg.lower():
                                validation_result["suggestions"].append("Query validation failed - check column references and data types")
                    
                    # Step 3: Extract table references if validation succeeded
                    if validation_result["is_valid"]:
                        # Extract table names using regex (basic approach)
                        table_patterns = [
                            r'\bFROM\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',  # FROM table
                            r'\bJOIN\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',  # JOIN table
                            r'\bUPDATE\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',  # UPDATE table
                            r'\bINTO\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',  # INSERT INTO table
                        ]
                        
                        tables = set()
                        for pattern in table_patterns:
                            matches = re.findall(pattern, query_stripped, re.IGNORECASE)
                            tables.update(match.strip('"\'`[]') for match in matches)
                        
                        validation_result["affected_tables"] = list(tables)
                        
                        # Add informational warnings
                        if len(tables) > 5:
                            validation_result["warnings"].append(f"Query involves {len(tables)} tables - consider query complexity")
                        
                except Exception as conn_error:
                    validation_result["error"] = f"Database connection error during validation: {str(conn_error)}"
                    validation_result["error_type"] = "connection_error"
                    
        except Exception as e:
            validation_result["error"] = f"Validation system error: {str(e)}"
            validation_result["error_type"] = "internal_error"
            logger.error(f"SQL validation error: {e}")
            
        return validation_result
    
    def execute_sql_query(self, sql_query: str, limit: int = 1000) -> Dict[str, Any]:
        """Execute a validated SQL query and return results safely.
        
        Args:
            sql_query: SQL query to execute (must pass validation first)
            limit: Maximum number of rows to return (safety limit)
            
        Returns:
            Dictionary with query results and execution metadata
        """
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        # Validate and cap the limit
        if limit <= 0 or limit > 5000:
            limit = min(max(limit, 100), 5000)
            
        result_data = {
            "success": False,
            "data": [],
            "columns": [],
            "row_count": 0,
            "execution_time_ms": None,
            "error": None,
            "error_type": None,
            "warnings": [],
            "query_plan": None,
            "limit_applied": False
        }
        
        try:
            import time
            start_time = time.time()
            
            # Step 1: Mandatory validation
            validation = self.validate_sql_syntax(sql_query)
            if not validation["is_valid"]:
                result_data["error"] = validation["error"]
                result_data["error_type"] = validation["error_type"]
                result_data["database_error"] = validation.get("database_error")
                return result_data
            
            # Transfer warnings from validation
            result_data["warnings"].extend(validation.get("warnings", []))
            
            # Step 2: Apply safety limits
            query_to_execute = sql_query.strip().rstrip(';')
            query_upper = query_to_execute.upper()
            
            # Add LIMIT if not present and it's a data-returning query
            needs_limit = (validation["query_type"] in ["SELECT", "CTE_SELECT"] and 
                         "LIMIT" not in query_upper and 
                         "TOP " not in query_upper)  # Snowflake also supports TOP
            
            if needs_limit:
                query_to_execute = f"{query_to_execute} LIMIT {limit}"
                result_data["limit_applied"] = True
                result_data["warnings"].append(f"Safety LIMIT {limit} applied to prevent large result sets")
            
            # Step 3: Execute the query
            with self.get_connection() as conn:
                result = conn.execute(text(query_to_execute))
                
                if result.returns_rows:
                    # Handle SELECT-type queries
                    result_data["columns"] = list(result.keys())
                    
                    rows = []
                    row_count = 0
                    for row in result.fetchall():
                        row_count += 1
                        row_dict = {}
                        for i, value in enumerate(row):
                            column_name = result_data["columns"][i]
                            # Serialize complex data types
                            if value is not None:
                                if hasattr(value, 'isoformat'):  # datetime/date objects
                                    row_dict[column_name] = value.isoformat()
                                elif isinstance(value, (bytes, bytearray)):  # binary data
                                    row_dict[column_name] = f"<binary:{len(value)} bytes>"
                                elif isinstance(value, (dict, list)):  # JSON/array types
                                    row_dict[column_name] = value
                                elif hasattr(value, '__dict__'):  # Complex objects
                                    row_dict[column_name] = str(value)
                                else:
                                    row_dict[column_name] = value
                            else:
                                row_dict[column_name] = None
                        rows.append(row_dict)
                    
                    result_data["data"] = rows
                    result_data["row_count"] = row_count
                    
                    if row_count == limit and result_data["limit_applied"]:
                        result_data["warnings"].append(f"Result set may be truncated at {limit} rows")
                        
                else:
                    # Handle non-returning queries (EXPLAIN, etc.)
                    result_data["row_count"] = getattr(result, 'rowcount', 0)
                
                end_time = time.time()
                result_data["execution_time_ms"] = round((end_time - start_time) * 1000, 2)
                result_data["success"] = True
                
                logger.info(f"SQL query executed: {result_data['row_count']} rows in {result_data['execution_time_ms']}ms")
                
        except (SQLAlchemyError, ProgrammingError) as e:
            result_data["error"] = str(e)
            result_data["error_type"] = "execution_error"
            logger.error(f"SQL execution failed: {e}")
        except Exception as e:
            result_data["error"] = f"Unexpected execution error: {str(e)}"
            result_data["error_type"] = "internal_error"
            logger.error(f"Unexpected SQL execution error: {e}")
            
        return result_data
    
    def has_engine(self) -> bool:
        """Check if database engine exists (basic connection check)."""
        return self.engine is not None
    
    def restore_connection_if_needed(self) -> bool:
        """Attempt to restore connection if engine is missing but params are available."""
        if not self.has_engine() and self._last_connection_params:
            logger.info("restore_connection_if_needed: Attempting to restore connection")
            try:
                self._reconnect()
                return True
            except Exception as e:
                logger.error(f"restore_connection_if_needed: Failed to restore connection: {e}")
                return False
        return self.has_engine()
    
    def force_reconnect(self) -> bool:
        """Force a reconnection even if engine exists (for troubleshooting)."""
        if not self._last_connection_params:
            logger.error("force_reconnect: No connection parameters available")
            return False
        
        logger.info("force_reconnect: Forcing database reconnection")
        try:
            # Clear current connection
            if self.engine:
                self.engine.dispose()
                self.engine = None
                logger.debug("force_reconnect: Disposed existing engine")
            
            # Reconnect
            self._reconnect()
            logger.info("force_reconnect: Reconnection successful")
            return True
        except Exception as e:
            logger.error(f"force_reconnect: Failed to reconnect: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if database is currently connected and healthy."""
        return self.has_engine() and self._test_connection()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status information."""
        if not self.engine:
            return {
                "connected": False,
                "connection_info": None,
                "last_params_available": self._last_connection_params is not None
            }
        
        is_healthy = self._test_connection()
        return {
            "connected": is_healthy,
            "connection_info": self.connection_info.copy(),
            "last_params_available": self._last_connection_params is not None,
            "engine_pool_size": self.engine.pool.size() if hasattr(self.engine, 'pool') else None,
            "engine_checked_out": self.engine.pool.checkedout() if hasattr(self.engine, 'pool') else None
        }
    
    def disconnect(self):
        """Close the database connection and clear stored parameters."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.metadata = None
            self.connection_info = {}
            self._last_connection_params = None
            logger.info("Database connection closed and parameters cleared")
