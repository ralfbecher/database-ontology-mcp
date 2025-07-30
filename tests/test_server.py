"""Tests for the database ontology MCP server."""

import asyncio
import unittest
from unittest.mock import patch, MagicMock, Mock
import json

import src.main as main_module
from src.database_manager import TableInfo, ColumnInfo


class TestMCPTools(unittest.TestCase):
    """Test suite for MCP tool functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_table_info = TableInfo(
            name="users",
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
                    name="name",
                    data_type="VARCHAR(255)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="email",
                    data_type="VARCHAR(255)",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["id"],
            foreign_keys=[],
            row_count=150
        )

    @patch('src.main.db_manager')
    def test_connect_database_postgresql_success(self, mock_db_manager):
        """Test successful PostgreSQL connection."""
        mock_db_manager.connect_postgresql.return_value = True
        
        result = main_module.connect_database.fn(
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

    @patch('src.main.db_manager')
    def test_connect_database_postgresql_failure(self, mock_db_manager):
        """Test PostgreSQL connection failure."""
        mock_db_manager.connect_postgresql.return_value = False
        
        result = main_module.connect_database.fn(
            db_type="postgresql",
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="wrongpass"
        )
        
        self.assertIn("Failed to connect", result)
        self.assertIn("postgresql", result)

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
        """Test connection with unsupported database type."""
        result = main_module.connect_database.fn(db_type="mysql")
        
        self.assertIn("Unsupported database type", result)
        self.assertIn("mysql", result)

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

    @patch('src.main.db_manager')
    def test_list_schemas_success(self, mock_db_manager):
        """Test successful schema listing."""
        mock_db_manager.get_schemas.return_value = ["public", "private", "analytics"]
        
        result = list_schemas()
        
        self.assertEqual(len(result), 3)
        self.assertIn("public", result)
        self.assertIn("private", result)
        self.assertIn("analytics", result)

    @patch('src.main.db_manager')
    def test_list_schemas_no_connection(self, mock_db_manager):
        """Test schema listing without connection."""
        mock_db_manager.get_schemas.side_effect = RuntimeError("No database connection")
        
        result = list_schemas()
        
        self.assertEqual(len(result), 1)
        self.assertIn("Error:", result[0])

    @patch('src.main.db_manager')
    def test_analyze_schema_success(self, mock_db_manager):
        """Test successful schema analysis."""
        mock_db_manager.get_tables.return_value = ["users", "orders"]
        mock_db_manager.analyze_table.side_effect = [
            self.sample_table_info,
            TableInfo(
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
                    )
                ],
                primary_keys=["id"],
                foreign_keys=[{"column": "user_id", "referenced_table": "users", "referenced_column": "id"}],
                row_count=500
            )
        ]
        
        result = analyze_schema("public")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["schema"], "public")
        self.assertEqual(result["table_count"], 2)
        self.assertEqual(len(result["tables"]), 2)
        
        # Check first table
        users_table = result["tables"][0]
        self.assertEqual(users_table["name"], "users")
        self.assertEqual(len(users_table["columns"]), 3)
        self.assertEqual(users_table["primary_keys"], ["id"])
        self.assertEqual(users_table["row_count"], 150)

    @patch('src.main.db_manager')
    def test_analyze_schema_no_connection(self, mock_db_manager):
        """Test schema analysis without connection."""
        mock_db_manager.get_tables.side_effect = RuntimeError("No database connection")
        
        result = analyze_schema("public")
        
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

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
    def test_sample_table_data_success(self, mock_db_manager):
        """Test successful table data sampling."""
        sample_data = [
            {"id": 1, "name": "John Doe", "email": "john@example.com"},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
        ]
        mock_db_manager.sample_table_data.return_value = sample_data
        
        result = sample_table_data("users", "public", 10)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "John Doe")
        self.assertEqual(result[1]["email"], "jane@example.com")

    @patch('src.main.db_manager')
    def test_sample_table_data_no_connection(self, mock_db_manager):
        """Test table data sampling without connection."""
        mock_db_manager.sample_table_data.side_effect = RuntimeError("No database connection")
        
        result = sample_table_data("users", "public", 10)
        
        self.assertEqual(len(result), 1)
        self.assertIn("error", result[0])

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
    """Test suite for ontology generation functionality."""

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
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="name",
                    data_type="VARCHAR(100)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["id"],
            foreign_keys=[],
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
        """Test name cleaning for URI generation."""
        self.assertEqual(self.generator._clean_name("test_table"), "test_table")
        self.assertEqual(self.generator._clean_name("test-table"), "test_table")
        self.assertEqual(self.generator._clean_name("test table"), "test_table")
        self.assertEqual(self.generator._clean_name("123test"), "_123test")
        self.assertEqual(self.generator._clean_name(""), "unnamed")


if __name__ == '__main__':
    unittest.main()

