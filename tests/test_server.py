"""Comprehensive tests for the enhanced database ontology MCP server."""

import json
import pytest
import unittest
from unittest.mock import patch, MagicMock, Mock, PropertyMock
from concurrent.futures import Future

import src.main as main_module
from src.database_manager import DatabaseManager, TableInfo, ColumnInfo
from src.ontology_generator import OntologyGenerator
from src.config import ConfigManager
from src.constants import SUPPORTED_DB_TYPES


class TestMCPTools(unittest.TestCase):
    """Comprehensive test suite for MCP tool functions with enhanced coverage."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample table info for users table
        self.sample_users_table = TableInfo(
            name="users",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False,
                    comment="User unique identifier"
                ),
                ColumnInfo(
                    name="name",
                    data_type="VARCHAR(255)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="User full name"
                ),
                ColumnInfo(
                    name="email",
                    data_type="VARCHAR(255)",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="User email address"
                )
            ],
            primary_keys=["id"],
            foreign_keys=[],
            comment="User accounts table",
            row_count=150
        )
        
        # Sample table info for orders table with foreign key
        self.sample_orders_table = TableInfo(
            name="orders",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="user_id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=True,
                    foreign_key_table="users",
                    foreign_key_column="id"
                ),
                ColumnInfo(
                    name="total_amount",
                    data_type="DECIMAL(10,2)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["id"],
            foreign_keys=[
                {
                    "column": "user_id",
                    "referenced_table": "users",
                    "referenced_column": "id"
                }
            ],
            row_count=500
        )

    @patch('src.main._server_state')
    def test_connect_database_postgresql_success(self, mock_server_state):
        """Test successful PostgreSQL connection with enhanced validation."""
        mock_db_manager = Mock()
        mock_db_manager.connect_postgresql.return_value = True
        mock_server_state.get_db_manager.return_value = mock_db_manager
        
        with patch('src.main.config_manager') as mock_config_manager:
            mock_config_manager.validate_db_config.return_value = {
                "valid": False,
                "missing_params": []
            }
            
            result = main_module.connect_database(
                db_type="postgresql",
                host="localhost",
                port=5432,
                database="testdb",
                username="testuser",
                password="testpass"
            )
        
        self.assertIn("Successfully connected", result)
        self.assertIn("postgresql", result)
        self.assertIn("testdb", result)
        mock_db_manager.connect_postgresql.assert_called_once_with(
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass"
        )

    @patch('src.main._server_state')
    def test_connect_database_postgresql_failure(self, mock_server_state):
        """Test PostgreSQL connection failure with proper error handling."""
        mock_db_manager = Mock()
        mock_db_manager.connect_postgresql.return_value = False
        mock_server_state.get_db_manager.return_value = mock_db_manager
        
        with patch('src.main.config_manager') as mock_config_manager:
            mock_config_manager.validate_db_config.return_value = {
                "valid": False,
                "missing_params": []
            }
            
            result = main_module.connect_database(
                db_type="postgresql",
                host="localhost",
                port=5432,
                database="testdb",
                username="testuser",
                password="wrongpass"
            )
        
        # Should return error JSON
        error_data = json.loads(result)
        self.assertEqual(error_data["error_type"], "connection_error")
        self.assertIn("Failed to connect", error_data["error"])

    @patch('src.main.db_manager')
    def test_connect_database_snowflake_success(self, mock_db_manager):
        """Test successful Snowflake connection."""
        mock_db_manager.connect_snowflake.return_value = True
        
        result = main_module.connect_database.fn(
            db_type="snowflake",
            account="test-account",
            username="testuser",
            password="testpass",
            warehouse="COMPUTE_WH",
            database="TESTDB",
            schema="PUBLIC"
        )
        
        self.assertIn("Successfully connected", result)
        self.assertIn("snowflake", result)

    def test_connect_database_unsupported_type(self):
        """Test connection with unsupported database type returns proper validation error."""
        result = main_module.connect_database(db_type="mysql")
        
        error_data = json.loads(result)
        self.assertEqual(error_data["error_type"], "validation_error")
        self.assertIn("Invalid database type", error_data["error"])
        self.assertIn("mysql", error_data["error"])
    
    def test_connect_database_missing_parameters(self):
        """Test connection with missing required parameters."""
        result = main_module.connect_database(
            db_type="postgresql",
            host="localhost"
            # Missing required parameters
        )
        
        error_data = json.loads(result)
        self.assertEqual(error_data["error_type"], "validation_error")
        self.assertIn("Missing required parameters", error_data["error"])

    @patch('src.main.db_manager')
    def test_connect_database_exception(self, mock_db_manager):
        """Test connection with exception."""
        mock_db_manager.connect_postgresql.side_effect = Exception("Connection error")
        
        result = main_module.connect_database.fn(
            db_type="postgresql",
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        
        self.assertIn("Connection error", result)

    @patch('src.main._server_state')
    def test_list_schemas_success(self, mock_server_state):
        """Test successful schema listing with enhanced validation."""
        mock_db_manager = Mock()
        mock_db_manager.get_schemas.return_value = ["public", "private", "analytics"]
        mock_server_state.get_db_manager.return_value = mock_db_manager
        
        result = main_module.list_schemas()
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIn("public", result)
        self.assertIn("private", result)
        self.assertIn("analytics", result)

    @patch('src.main._server_state')
    def test_list_schemas_no_connection(self, mock_server_state):
        """Test schema listing without connection returns proper error."""
        mock_db_manager = Mock()
        mock_db_manager.get_schemas.side_effect = RuntimeError("No database connection")
        mock_server_state.get_db_manager.return_value = mock_db_manager
        
        result = main_module.list_schemas()
        
        error_data = json.loads(result)
        self.assertEqual(error_data["error_type"], "connection_error")
        self.assertIn("No database connection", error_data["error"])

    @patch('src.main._server_state')
    def test_analyze_schema_success(self, mock_server_state):
        """Test successful schema analysis with parallel processing."""
        mock_db_manager = Mock()
        mock_db_manager.get_tables.return_value = ["users", "orders"]
        mock_db_manager.analyze_table.side_effect = [
            self.sample_users_table,
            self.sample_orders_table
        ]
        mock_server_state.get_db_manager.return_value = mock_db_manager
        
        # Mock thread pool for testing
        mock_thread_pool = Mock()
        mock_future1 = Mock()
        mock_future1.result.return_value = self.sample_users_table
        mock_future2 = Mock()
        mock_future2.result.return_value = self.sample_orders_table
        
        mock_thread_pool.__enter__.return_value = mock_thread_pool
        mock_thread_pool.__exit__.return_value = None
        mock_thread_pool.submit.side_effect = [mock_future1, mock_future2]
        mock_server_state.thread_pool = mock_thread_pool
        
        with patch('src.main.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2]
            
            result = main_module.analyze_schema("public")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["schema"], "public")
        self.assertEqual(result["table_count"], 2)
        self.assertEqual(len(result["tables"]), 2)
        
        # Check users table structure
        users_table = next(t for t in result["tables"] if t["name"] == "users")
        self.assertEqual(len(users_table["columns"]), 3)
        self.assertEqual(users_table["primary_keys"], ["id"])
        self.assertEqual(users_table["row_count"], 150)
        
        # Check orders table with foreign key
        orders_table = next(t for t in result["tables"] if t["name"] == "orders")
        self.assertEqual(len(orders_table["foreign_keys"]), 1)
        self.assertEqual(orders_table["foreign_keys"][0]["referenced_table"], "users")

    @patch('src.main._server_state')
    def test_analyze_schema_no_connection(self, mock_server_state):
        """Test schema analysis without connection returns proper error."""
        mock_db_manager = Mock()
        mock_db_manager.get_tables.side_effect = RuntimeError("No database connection")
        mock_server_state.get_db_manager.return_value = mock_db_manager
        
        result = main_module.analyze_schema("public")
        
        error_data = json.loads(result)
        self.assertEqual(error_data["error_type"], "connection_error")
        self.assertIn("No database connection", error_data["error"])

    @patch('src.main.db_manager')
    @patch('src.main.ontology_generator')
    def test_generate_ontology_success(self, mock_ontology_generator, mock_db_manager):
        """Test successful ontology generation."""
        mock_db_manager.get_tables.return_value = ["users"]
        mock_db_manager.analyze_table.return_value = self.sample_table_info
        
        # Mock the ontology generator
        mock_generator_instance = Mock()
        mock_generator_instance.generate_from_schema.return_value = "@prefix ex: <http://example.com/ontology/> ."
        mock_ontology_generator.return_value = mock_generator_instance
        
        # Mock the global ontology_generator
        with patch('src.main.OntologyGenerator') as mock_class:
            mock_class.return_value = mock_generator_instance
            
            result = main_module.generate_ontology.fn("public", "http://example.com/ontology/", False)
        
        self.assertIsInstance(result, str)
        self.assertIn("@prefix ex:", result)

    @patch('src.main.db_manager')
    def test_generate_ontology_no_tables(self, mock_db_manager):
        """Test ontology generation with no tables."""
        mock_db_manager.get_tables.return_value = []
        
        result = main_module.generate_ontology.fn("public")
        
        self.assertIn("Error: No tables found", result)

    @patch('src.main.db_manager')
    @patch('src.main._server_state')
    def test_sample_table_data_success(self, mock_server_state):
        """Test successful table data sampling with validation."""
        mock_db_manager = Mock()
        sample_data = [
            {"id": 1, "name": "John Doe", "email": "john@example.com"},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
        ]
        mock_db_manager.sample_table_data.return_value = sample_data
        mock_server_state.get_db_manager.return_value = mock_db_manager
        
        result = main_module.sample_table_data("users", "public", 10)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "John Doe")
        self.assertEqual(result[1]["email"], "jane@example.com")
        
        # Verify the call was made with correct parameters
        mock_db_manager.sample_table_data.assert_called_once_with("users", "public", 10)

    def test_sample_table_data_invalid_table_name(self):
        """Test table data sampling with invalid table name."""
        result = main_module.sample_table_data("", "public", 10)
        
        error_data = json.loads(result)
        self.assertEqual(error_data["error_type"], "validation_error")
        self.assertIn("Table name is required", error_data["error"])
    
    @patch('src.main._server_state')
    def test_sample_table_data_database_error(self, mock_server_state):
        """Test table data sampling with database error."""
        mock_db_manager = Mock()
        mock_db_manager.sample_table_data.side_effect = ValueError("Invalid table name format")
        mock_server_state.get_db_manager.return_value = mock_db_manager
        
        result = main_module.sample_table_data("invalid-table", "public", 10)
        
        error_data = json.loads(result)
        self.assertEqual(error_data["error_type"], "validation_error")
        self.assertIn("Invalid table name format", error_data["error"])

    @patch('src.main.db_manager')
    def test_get_table_relationships_success(self, mock_db_manager):
        """Test successful table relationship retrieval."""
        relationships = {
            "orders": [
                {"column": "user_id", "referenced_table": "users", "referenced_column": "id"}
            ],
            "order_items": [
                {"column": "order_id", "referenced_table": "orders", "referenced_column": "id"}
            ]
        }
        mock_db_manager.get_table_relationships.return_value = relationships
        
        result = get_table_relationships("public")
        
        self.assertIsInstance(result, dict)
        self.assertIn("orders", result)
        self.assertIn("order_items", result)
        self.assertEqual(len(result["orders"]), 1)
        self.assertEqual(result["orders"][0]["referenced_table"], "users")

    @patch('src.main.db_manager')
    def test_get_table_relationships_no_connection(self, mock_db_manager):
        """Test table relationships without connection."""
        mock_db_manager.get_table_relationships.side_effect = RuntimeError("No database connection")
        
        result = get_table_relationships("public")
        
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)


class TestOntologyGenerator(unittest.TestCase):
    """Enhanced test suite for ontology generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from src.ontology_generator import OntologyGenerator
        self.generator = OntologyGenerator()
        
        self.sample_table = TableInfo(
            name="test_table",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False,
                    comment="Primary key"
                ),
                ColumnInfo(
                    name="name",
                    data_type="VARCHAR(100)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="Entity name"
                ),
                ColumnInfo(
                    name="created_at",
                    data_type="TIMESTAMP",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="Creation timestamp"
                )
            ],
            primary_keys=["id"],
            foreign_keys=[],
            comment="Test table for ontology generation",
            row_count=10
        )

    def test_generate_ontology_structure(self):
        """Test that generated ontology has proper structure."""
        result = self.generator.generate_from_schema([self.sample_table])
        
        self.assertIn("@prefix ex:", result)
        self.assertIn("@prefix owl:", result)
        self.assertIn("@prefix rdfs:", result)
        self.assertIn("@prefix xsd:", result)
        self.assertIn("owl:Class", result)
        self.assertIn("owl:DatatypeProperty", result)

    def test_xsd_type_mapping(self):
        """Test XSD type mapping for various SQL types."""
        from rdflib.namespace import XSD
        
        # Test integer mapping
        self.assertEqual(self.generator._map_sql_to_xsd("INTEGER"), XSD.integer)
        self.assertEqual(self.generator._map_sql_to_xsd("BIGINT"), XSD.integer)
        
        # Test string mapping
        self.assertEqual(self.generator._map_sql_to_xsd("VARCHAR(255)"), XSD.string)
        self.assertEqual(self.generator._map_sql_to_xsd("TEXT"), XSD.string)
        
        # Test boolean mapping
        self.assertEqual(self.generator._map_sql_to_xsd("BOOLEAN"), XSD.boolean)
        
        # Test datetime mapping
        self.assertEqual(self.generator._map_sql_to_xsd("TIMESTAMP"), XSD.dateTime)
        self.assertEqual(self.generator._map_sql_to_xsd("DATE"), XSD.date)
        
        # Test numeric mapping
        self.assertEqual(self.generator._map_sql_to_xsd("DECIMAL(10,2)"), XSD.decimal)
        self.assertEqual(self.generator._map_sql_to_xsd("FLOAT"), XSD.float)

    def test_clean_name_function(self):
        """Test name cleaning for URI generation with edge cases."""
        self.assertEqual(self.generator._clean_name("test_table"), "test_table")
        self.assertEqual(self.generator._clean_name("test-table"), "test_table")
        self.assertEqual(self.generator._clean_name("test table"), "test_table")
        self.assertEqual(self.generator._clean_name("123test"), "_123test")
        self.assertEqual(self.generator._clean_name(""), "unnamed")
        self.assertEqual(self.generator._clean_name("test@table#123"), "test_table_123")
        self.assertEqual(self.generator._clean_name("test.table"), "test_table")
    
    def test_enrichment_data_generation(self):
        """Test generation of enrichment data structure."""
        sample_data = {
            "test_table": [
                {"id": 1, "name": "Test Item 1", "created_at": "2023-01-01T00:00:00"},
                {"id": 2, "name": "Test Item 2", "created_at": "2023-01-02T00:00:00"}
            ]
        }
        
        enrichment_data = self.generator.get_enrichment_data([self.sample_table], sample_data)
        
        self.assertIn("schema_data", enrichment_data)
        self.assertIn("instructions", enrichment_data)
        
        schema_data = enrichment_data["schema_data"]
        self.assertEqual(len(schema_data), 1)
        
        table_data = schema_data[0]
        self.assertEqual(table_data["table_name"], "test_table")
        self.assertEqual(len(table_data["columns"]), 3)
        self.assertIn("sample_data", table_data)
        self.assertEqual(len(table_data["sample_data"]), 2)
    
    def test_apply_enrichment(self):
        """Test application of enrichment suggestions."""
        # Generate base ontology
        ontology_ttl = self.generator.generate_from_schema([self.sample_table])
        
        # Define enrichment suggestions
        enrichment_suggestions = {
            "classes": [
                {
                    "original_name": "test_table",
                    "suggested_name": "TestEntity",
                    "description": "A test entity for demonstration purposes"
                }
            ],
            "properties": [
                {
                    "table_name": "test_table",
                    "original_name": "name",
                    "suggested_name": "entityName",
                    "description": "The name of the test entity"
                }
            ],
            "relationships": []
        }
        
        # Apply enrichment
        self.generator.apply_enrichment(enrichment_suggestions)
        
        # Serialize and check result
        enriched_ontology = self.generator.serialize_ontology()
        self.assertIn("TestEntity", enriched_ontology)
        self.assertIn("entityName", enriched_ontology)


class TestConfigManager(unittest.TestCase):
    """Test suite for configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.config import ConfigManager
        self.config_manager = ConfigManager()
    
    def test_validate_db_config_postgresql(self):
        """Test PostgreSQL configuration validation."""
        with patch.dict('os.environ', {
            'POSTGRES_HOST': 'localhost',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DATABASE': 'testdb',
            'POSTGRES_USERNAME': 'testuser',
            'POSTGRES_PASSWORD': 'testpass'
        }):
            validation = self.config_manager.validate_db_config('postgresql')
            self.assertTrue(validation['valid'])
            self.assertEqual(len(validation['missing_params']), 0)
    
    def test_validate_db_config_missing_params(self):
        """Test configuration validation with missing parameters."""
        with patch.dict('os.environ', {}, clear=True):
            validation = self.config_manager.validate_db_config('postgresql')
            self.assertFalse(validation['valid'])
            self.assertGreater(len(validation['missing_params']), 0)
    
    def test_validate_db_config_invalid_type(self):
        """Test configuration validation with invalid database type."""
        with self.assertRaises(ValueError):
            self.config_manager.validate_db_config('invalid_db_type')


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions."""
    
    def test_sanitize_for_logging(self):
        """Test sanitization of sensitive data for logging."""
        from src.utils import sanitize_for_logging
        
        sensitive_data = {
            'host': 'localhost',
            'port': 5432,
            'password': 'secret123',
            'api_key': 'sk-1234567890',
            'config': {
                'username': 'user',
                'secret': 'hidden'
            }
        }
        
        sanitized = sanitize_for_logging(sensitive_data)
        
        self.assertEqual(sanitized['host'], 'localhost')
        self.assertEqual(sanitized['port'], 5432)
        self.assertEqual(sanitized['password'], '***REDACTED***')
        self.assertEqual(sanitized['api_key'], '***REDACTED***')
        self.assertEqual(sanitized['config']['username'], 'user')
        self.assertEqual(sanitized['config']['secret'], '***REDACTED***')
    
    def test_validate_uri(self):
        """Test URI validation function."""
        from src.utils import validate_uri
        
        self.assertTrue(validate_uri('https://example.com/'))
        self.assertTrue(validate_uri('http://localhost:8080/'))
        self.assertTrue(validate_uri('https://api.example.com/v1/ontology/'))
        
        self.assertFalse(validate_uri(''))
        self.assertFalse(validate_uri('not-a-uri'))
        self.assertFalse(validate_uri('ftp://example.com/'))
    
    def test_format_bytes(self):
        """Test bytes formatting function."""
        from src.utils import format_bytes
        
        self.assertEqual(format_bytes(0), "0 B")
        self.assertEqual(format_bytes(1024), "1.0 KB")
        self.assertEqual(format_bytes(1024 * 1024), "1.0 MB")
        self.assertEqual(format_bytes(1536), "1.5 KB")


if __name__ == '__main__':
    # Use pytest for better test discovery and reporting if available
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        unittest.main(verbosity=2)

