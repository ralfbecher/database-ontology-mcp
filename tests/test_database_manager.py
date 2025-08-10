"""Tests for the DatabaseManager class with enhanced coverage."""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from src.database_manager import DatabaseManager, TableInfo, ColumnInfo
from src.constants import POSTGRES_SYSTEM_SCHEMAS, SNOWFLAKE_SYSTEM_SCHEMAS


class TestDatabaseManager(unittest.TestCase):
    """Test suite for DatabaseManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db_manager = DatabaseManager()
    
    def test_init(self):
        """Test DatabaseManager initialization."""
        self.assertIsNone(self.db_manager.engine)
        self.assertIsNone(self.db_manager.metadata)
        self.assertEqual(self.db_manager.connection_info, {})
        self.assertEqual(self.db_manager._connection_pool_size, 5)
        self.assertEqual(self.db_manager._max_overflow, 10)
    
    @patch('src.database_manager.create_engine')
    def test_connect_postgresql_success(self, mock_create_engine):
        """Test successful PostgreSQL connection."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Mock the connection context manager
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        
        result = self.db_manager.connect_postgresql(
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        
        self.assertTrue(result)
        self.assertIsNotNone(self.db_manager.engine)
        self.assertEqual(self.db_manager.connection_info["type"], "postgresql")
        self.assertEqual(self.db_manager.connection_info["host"], "localhost")
        self.assertEqual(self.db_manager.connection_info["database"], "testdb")
    
    @patch('src.database_manager.create_engine')
    def test_connect_postgresql_failure(self, mock_create_engine):
        """Test PostgreSQL connection failure."""
        mock_create_engine.side_effect = SQLAlchemyError("Connection failed")
        
        result = self.db_manager.connect_postgresql(
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="wrongpass"
        )
        
        self.assertFalse(result)
        self.assertIsNone(self.db_manager.engine)
    
    def test_connect_postgresql_missing_params(self):
        """Test PostgreSQL connection with missing parameters."""
        result = self.db_manager.connect_postgresql(
            host="",  # Missing host
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        
        self.assertFalse(result)
    
    @patch('src.database_manager.create_engine')
    def test_connect_snowflake_success(self, mock_create_engine):
        """Test successful Snowflake connection."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Mock the connection context manager
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        
        result = self.db_manager.connect_snowflake(
            account="test-account",
            username="testuser",
            password="testpass",
            warehouse="COMPUTE_WH",
            database="TESTDB",
            schema="PUBLIC"
        )
        
        self.assertTrue(result)
        self.assertIsNotNone(self.db_manager.engine)
        self.assertEqual(self.db_manager.connection_info["type"], "snowflake")
        self.assertEqual(self.db_manager.connection_info["account"], "test-account")
    
    def test_validate_identifier(self):
        """Test identifier validation."""
        # Valid identifiers
        self.assertTrue(self.db_manager._validate_identifier("valid_table"))
        self.assertTrue(self.db_manager._validate_identifier("table123"))
        self.assertTrue(self.db_manager._validate_identifier("_table"))
        self.assertTrue(self.db_manager._validate_identifier("table-name"))
        
        # Invalid identifiers
        self.assertFalse(self.db_manager._validate_identifier(""))
        self.assertFalse(self.db_manager._validate_identifier("123table"))
        self.assertFalse(self.db_manager._validate_identifier("table@name"))
        self.assertFalse(self.db_manager._validate_identifier("table name"))
        self.assertFalse(self.db_manager._validate_identifier("a" * 64))  # Too long
    
    @patch('src.database_manager.inspect')
    def test_analyze_table_success(self, mock_inspect):
        """Test successful table analysis."""
        # Setup mocks
        self.db_manager.engine = Mock()
        mock_inspector = Mock()
        mock_inspect.return_value = mock_inspector
        
        mock_inspector.has_table.return_value = True
        mock_inspector.get_columns.return_value = [
            {
                'name': 'id',
                'type': 'INTEGER',
                'nullable': False,
                'comment': 'Primary key'
            },
            {
                'name': 'name',
                'type': 'VARCHAR(255)',
                'nullable': False,
                'comment': 'User name'
            }
        ]
        mock_inspector.get_pk_constraint.return_value = {
            'constrained_columns': ['id']
        }
        mock_inspector.get_foreign_keys.return_value = []
        
        # Mock connection for row count
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 100
        mock_conn.execute.return_value = mock_result
        self.db_manager.engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        self.db_manager.engine.connect.return_value.__exit__ = Mock(return_value=None)
        
        result = self.db_manager.analyze_table("test_table", "public")
        
        self.assertIsInstance(result, TableInfo)
        self.assertEqual(result.name, "test_table")
        self.assertEqual(result.schema, "public")
        self.assertEqual(len(result.columns), 2)
        self.assertEqual(result.primary_keys, ['id'])
        self.assertEqual(result.row_count, 100)
    
    def test_analyze_table_no_engine(self):
        """Test table analysis without engine."""
        with self.assertRaises(RuntimeError):
            self.db_manager.analyze_table("test_table")
    
    @patch('src.database_manager.inspect')
    def test_analyze_table_not_found(self, mock_inspect):
        """Test analysis of non-existent table."""
        self.db_manager.engine = Mock()
        mock_inspector = Mock()
        mock_inspect.return_value = mock_inspector
        
        mock_inspector.has_table.return_value = False
        
        result = self.db_manager.analyze_table("nonexistent_table")
        
        self.assertIsNone(result)
    
    def test_sample_table_data_no_engine(self):
        """Test data sampling without engine."""
        with self.assertRaises(RuntimeError):
            self.db_manager.sample_table_data("test_table")
    
    def test_sample_table_data_invalid_table_name(self):
        """Test data sampling with invalid table name."""
        self.db_manager.engine = Mock()
        
        with self.assertRaises(ValueError):
            self.db_manager.sample_table_data("invalid@table")
    
    def test_sample_table_data_invalid_schema(self):
        """Test data sampling with invalid schema name."""
        self.db_manager.engine = Mock()
        
        with self.assertRaises(ValueError):
            self.db_manager.sample_table_data("table", "invalid schema")
    
    def test_sample_table_data_limit_validation(self):
        """Test data sampling with various limit values."""
        self.db_manager.engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.keys.return_value = ['id', 'name']
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result
        self.db_manager.engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        self.db_manager.engine.connect.return_value.__exit__ = Mock(return_value=None)
        
        # Test with negative limit
        result = self.db_manager.sample_table_data("test_table", limit=-1)
        self.assertEqual(result, [])
        
        # Test with very large limit
        result = self.db_manager.sample_table_data("test_table", limit=2000)
        # Should be capped at MAX_SAMPLE_LIMIT (1000)
        mock_conn.execute.assert_called()
        args = mock_conn.execute.call_args
        self.assertEqual(args[1]['limit'], 1000)
    
    def test_disconnect(self):
        """Test database disconnection."""
        mock_engine = Mock()
        self.db_manager.engine = mock_engine
        self.db_manager.connection_info = {"type": "postgresql"}
        
        self.db_manager.disconnect()
        
        mock_engine.dispose.assert_called_once()
        self.assertIsNone(self.db_manager.engine)
        self.assertEqual(self.db_manager.connection_info, {})
    
    def test_get_connection_context_manager(self):
        """Test connection context manager."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value = mock_conn
        self.db_manager.engine = mock_engine
        
        with self.db_manager.get_connection() as conn:
            self.assertEqual(conn, mock_conn)
        
        mock_conn.close.assert_called_once()
    
    def test_get_connection_no_engine(self):
        """Test connection context manager without engine."""
        with self.assertRaises(RuntimeError):
            with self.db_manager.get_connection():
                pass


class TestTableInfoDataClass(unittest.TestCase):
    """Test suite for TableInfo and ColumnInfo data classes."""
    
    def test_column_info_creation(self):
        """Test ColumnInfo creation and attributes."""
        col_info = ColumnInfo(
            name="test_column",
            data_type="VARCHAR(255)",
            is_nullable=True,
            is_primary_key=False,
            is_foreign_key=True,
            foreign_key_table="ref_table",
            foreign_key_column="ref_id",
            comment="Test column"
        )
        
        self.assertEqual(col_info.name, "test_column")
        self.assertEqual(col_info.data_type, "VARCHAR(255)")
        self.assertTrue(col_info.is_nullable)
        self.assertFalse(col_info.is_primary_key)
        self.assertTrue(col_info.is_foreign_key)
        self.assertEqual(col_info.foreign_key_table, "ref_table")
        self.assertEqual(col_info.foreign_key_column, "ref_id")
        self.assertEqual(col_info.comment, "Test column")
    
    def test_table_info_creation(self):
        """Test TableInfo creation and attributes."""
        columns = [
            ColumnInfo(
                name="id",
                data_type="INTEGER",
                is_nullable=False,
                is_primary_key=True,
                is_foreign_key=False
            )
        ]
        
        table_info = TableInfo(
            name="test_table",
            schema="public",
            columns=columns,
            primary_keys=["id"],
            foreign_keys=[],
            comment="Test table",
            row_count=100
        )
        
        self.assertEqual(table_info.name, "test_table")
        self.assertEqual(table_info.schema, "public")
        self.assertEqual(len(table_info.columns), 1)
        self.assertEqual(table_info.primary_keys, ["id"])
        self.assertEqual(table_info.foreign_keys, [])
        self.assertEqual(table_info.comment, "Test table")
        self.assertEqual(table_info.row_count, 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)