"""Ontology generator for creating RDF graphs from database schemas."""

import json
import logging
import os
import re
from typing import List, Dict, Any, Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

from .constants import DEFAULT_BASE_URI, ONTOLOGY_TITLE, ONTOLOGY_DESCRIPTION
from .database_manager import TableInfo, ColumnInfo

logger = logging.getLogger(__name__)

# Define namespaces
EX = Namespace("http://example.com/ontology/")

class OntologyGenerator:
    """Generates an ontology from a database schema with comprehensive database annotations."""

    def __init__(self, base_uri: str = DEFAULT_BASE_URI):
        self.graph = Graph()
        self.base_uri = Namespace(base_uri)
        
        # Define custom namespace for database-specific annotations
        self.db_ns = Namespace(f"{base_uri}db/")
        
        self.graph.bind("ns", self.base_uri)
        self.graph.bind("db", self.db_ns)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

    def generate_from_schema(self, tables_info: List[TableInfo]) -> str:
        """Generate an ontology from a list of table information."""
        # Add ontology metadata
        ontology_uri = self.base_uri[""]
        self.graph.add((ontology_uri, RDF.type, OWL.Ontology))
        self.graph.add((ontology_uri, RDFS.label, Literal(ONTOLOGY_TITLE)))
        self.graph.add((ontology_uri, RDFS.comment, Literal(ONTOLOGY_DESCRIPTION)))
        
        for table_info in tables_info:
            self._add_table_to_ontology(table_info)

        return self.graph.serialize(format="turtle")

    def _add_table_to_ontology(self, table_info: TableInfo):
        """Add a single table and its columns to the ontology with comprehensive database annotations."""
        # Create proper URI for table class
        table_uri = self.base_uri[self._clean_name(table_info.name)]

        # Define table as a class
        self.graph.add((table_uri, RDF.type, OWL.Class))
        self.graph.add((table_uri, RDFS.label, Literal(table_info.name)))
        
        # Add comprehensive database-specific annotations
        self.graph.add((table_uri, self.db_ns.tableName, Literal(table_info.name)))
        self.graph.add((table_uri, self.db_ns.schemaName, Literal(table_info.schema)))
        
        if table_info.row_count is not None:
            self.graph.add((table_uri, self.db_ns.rowCount, Literal(table_info.row_count)))
            
        if table_info.comment:
            self.graph.add((table_uri, RDFS.comment, Literal(table_info.comment)))
            
        # Add primary key information
        if table_info.primary_keys:
            for pk in table_info.primary_keys:
                self.graph.add((table_uri, self.db_ns.primaryKey, Literal(pk)))

        # Add columns as properties
        for column in table_info.columns:
            self._add_column_to_ontology(table_uri, column, table_info.name)

        # Define relationships
        for fk in table_info.foreign_keys:
            self._add_relationship_to_ontology(table_uri, fk, table_info.name)

    def _add_column_to_ontology(self, table_uri: URIRef, column: ColumnInfo, table_name: str):
        """Add a column as a data property to the ontology with comprehensive database annotations."""
        # Create proper property URI
        prop_name = f"{self._clean_name(table_name)}_{self._clean_name(column.name)}"
        prop_uri = self.base_uri[prop_name]

        # Foreign key columns need both data property (for the value) and object property (for relationship)
        # Always create the data property for the column
        self.graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        
        self.graph.add((prop_uri, RDFS.domain, table_uri))
        self.graph.add((prop_uri, RDFS.label, Literal(column.name)))
        
        # Add comprehensive database-specific annotations
        self.graph.add((prop_uri, self.db_ns.columnName, Literal(column.name)))
        self.graph.add((prop_uri, self.db_ns.tableName, Literal(table_name)))
        self.graph.add((prop_uri, self.db_ns.sqlDataType, Literal(column.data_type)))
        self.graph.add((prop_uri, self.db_ns.isNullable, Literal(column.is_nullable)))
        self.graph.add((prop_uri, self.db_ns.isPrimaryKey, Literal(column.is_primary_key)))
        self.graph.add((prop_uri, self.db_ns.isForeignKey, Literal(column.is_foreign_key)))
        
        # Add SQL query generation hints
        full_column_ref = f"{table_name}.{column.name}"
        self.graph.add((prop_uri, self.db_ns.sqlReference, Literal(full_column_ref)))
        
        # Map SQL data types to proper XSD types
        xsd_type = self._map_sql_to_xsd(column.data_type)
        if xsd_type:
            self.graph.add((prop_uri, RDFS.range, xsd_type))

        # Note: Primary key and nullability constraints are already captured
        # in the metadata annotations (db:isPrimaryKey, db:isNullable).
        # We don't create OWL restriction classes as that would incorrectly
        # make table classes subclasses of restrictions.

        if column.comment:
            self.graph.add((prop_uri, RDFS.comment, Literal(column.comment)))

    def _add_relationship_to_ontology(self, table_uri: URIRef, fk: Dict[str, str], table_name: str):
        """Add a foreign key relationship as an object property with comprehensive database annotations."""
        # Create descriptive relationship name
        rel_name = f"{self._clean_name(table_name)}_has_{self._clean_name(fk['referenced_table'])}"
        prop_uri = self.base_uri[rel_name]
        referenced_table_uri = self.base_uri[self._clean_name(fk['referenced_table'])]

        self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
        self.graph.add((prop_uri, RDFS.domain, table_uri))
        self.graph.add((prop_uri, RDFS.range, referenced_table_uri))
        self.graph.add((prop_uri, RDFS.label, Literal(f"{table_name} has {fk['referenced_table']}")))
        
        # Add comprehensive database-specific annotations for foreign keys
        self.graph.add((prop_uri, self.db_ns.foreignKeyColumn, Literal(fk['column'])))
        self.graph.add((prop_uri, self.db_ns.referencedTable, Literal(fk['referenced_table'])))
        self.graph.add((prop_uri, self.db_ns.referencedColumn, Literal(fk['referenced_column'])))
        
        # Add SQL join condition
        join_condition = f"{table_name}.{fk['column']} = {fk['referenced_table']}.{fk['referenced_column']}"
        self.graph.add((prop_uri, self.db_ns.sqlJoinCondition, Literal(join_condition)))
        
        # Add relationship type annotation
        self.graph.add((prop_uri, self.db_ns.relationshipType, Literal("many_to_one")))
        
        # Add inverse relationship
        inverse_rel_name = f"{self._clean_name(fk['referenced_table'])}_referenced_by_{self._clean_name(table_name)}"
        inverse_prop_uri = self.base_uri[inverse_rel_name]
        self.graph.add((inverse_prop_uri, RDF.type, OWL.ObjectProperty))
        self.graph.add((inverse_prop_uri, RDFS.domain, referenced_table_uri))
        self.graph.add((inverse_prop_uri, RDFS.range, table_uri))
        self.graph.add((inverse_prop_uri, RDFS.label, Literal(f"{fk['referenced_table']} referenced by {table_name}")))
        
        # Add database annotations for inverse relationship
        self.graph.add((inverse_prop_uri, self.db_ns.relationshipType, Literal("one_to_many")))
        
        # Link them as inverses
        self.graph.add((prop_uri, OWL.inverseOf, inverse_prop_uri))

    def _clean_name(self, name: str) -> str:
        """Clean a name to make it suitable for URIs."""
        if not name:
            return 'unnamed'
        
        # Replace spaces and special characters with underscores
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure it starts with a letter or underscore
        if cleaned and not (cleaned[0].isalpha() or cleaned[0] == '_'):
            cleaned = '_' + cleaned
        
        return cleaned or 'unnamed'

    def _map_sql_to_xsd(self, sql_type: str) -> Optional[URIRef]:
        """Map SQL data types to XSD Schema datatypes."""
        sql_type = sql_type.lower()

        # Integer types - check tinyint first before checking for "int"
        if "tinyint" in sql_type:
            return XSD.byte
        if any(t in sql_type for t in ["int", "serial", "bigint", "smallint"]):
            return XSD.integer
        
        # String types
        if any(t in sql_type for t in ["char", "text", "varchar", "string"]):
            return XSD.string
        if "clob" in sql_type or "blob" in sql_type:
            return XSD.string
            
        # Temporal types
        if "timestamp" in sql_type or "datetime" in sql_type:
            return XSD.dateTime
        if sql_type.startswith("date"):
            return XSD.date
        if sql_type.startswith("time"):
            return XSD.time
            
        # Numeric types
        if any(t in sql_type for t in ["float", "real"]):
            return XSD.float
        if any(t in sql_type for t in ["double", "double precision"]):
            return XSD.double
        if any(t in sql_type for t in ["decimal", "numeric", "money"]):
            return XSD.decimal
            
        # Boolean types
        if any(t in sql_type for t in ["bool", "boolean", "bit"]):
            return XSD.boolean
            
        # Binary types
        if any(t in sql_type for t in ["binary", "varbinary", "bytea"]):
            return XSD.base64Binary
            
        # UUID types
        if "uuid" in sql_type:
            return XSD.string
            
        # JSON types
        if any(t in sql_type for t in ["json", "jsonb"]):
            return XSD.string
        
        # Default to string for unknown types
        logger.warning(f"Unknown SQL type '{sql_type}', mapping to xsd:string")
        return XSD.string

    def apply_semantic_descriptions(self, descriptions: Dict[str, Any]):
        """Apply LLM-generated semantic descriptions to the ontology.
        
        This method allows applying rich, context-aware descriptions generated
        by the LLM through the generate_semantic_descriptions tool.
        
        Args:
            descriptions: Dictionary containing semantic descriptions for:
                - tables: Business descriptions for each table
                - columns: Business meanings for each column
                - relationships: Descriptions of foreign key relationships
        """
        logger.info("Applying LLM-generated semantic descriptions to ontology")
        
        # Apply table descriptions
        if "tables" in descriptions:
            for table_name, table_desc in descriptions["tables"].items():
                table_uri = self.base_uri[self._clean_name(table_name)]
                if (table_uri, RDF.type, OWL.Class) in self.graph:
                    if "business_description" in table_desc:
                        # Remove existing basic description if present
                        self.graph.remove((table_uri, self.db_ns.businessDescription, None))
                        # Add new rich description
                        self.graph.add((table_uri, self.db_ns.businessDescription, 
                                      Literal(table_desc["business_description"])))
                    
                    if "table_type" in table_desc:
                        self.graph.add((table_uri, self.db_ns.tableType, 
                                      Literal(table_desc["table_type"])))
                    
                    if "usage_notes" in table_desc:
                        self.graph.add((table_uri, self.db_ns.usageNotes, 
                                      Literal(table_desc["usage_notes"])))
        
        # Apply column descriptions
        if "columns" in descriptions:
            for column_ref, column_desc in descriptions["columns"].items():
                # Parse table.column format
                if "." in column_ref:
                    table_name, column_name = column_ref.split(".", 1)
                    prop_name = f"{self._clean_name(table_name)}_{self._clean_name(column_name)}"
                    prop_uri = self.base_uri[prop_name]
                    
                    if ((prop_uri, RDF.type, OWL.DatatypeProperty) in self.graph or
                        (prop_uri, RDF.type, OWL.ObjectProperty) in self.graph):
                        
                        if "business_description" in column_desc:
                            # Remove existing basic description if present
                            self.graph.remove((prop_uri, self.db_ns.businessDescription, None))
                            # Add new rich description
                            self.graph.add((prop_uri, self.db_ns.businessDescription, 
                                          Literal(column_desc["business_description"])))
                        
                        if "data_characteristics" in column_desc:
                            self.graph.add((prop_uri, self.db_ns.dataCharacteristics, 
                                          Literal(column_desc["data_characteristics"])))
                        
                        if "business_rules" in column_desc:
                            self.graph.add((prop_uri, self.db_ns.businessRules, 
                                          Literal(column_desc["business_rules"])))
        
        # Apply relationship descriptions
        if "relationships" in descriptions:
            for rel_key, rel_desc in descriptions["relationships"].items():
                # Parse relationship key format
                # Expected format: "from_table.column -> to_table.column"
                if " -> " in rel_key:
                    from_part, to_part = rel_key.split(" -> ")
                    from_table = from_part.split(".")[0] if "." in from_part else from_part
                    to_table = to_part.split(".")[0] if "." in to_part else to_part
                    
                    rel_name = f"{self._clean_name(from_table)}_has_{self._clean_name(to_table)}"
                    rel_uri = self.base_uri[rel_name]
                    
                    if (rel_uri, RDF.type, OWL.ObjectProperty) in self.graph:
                        if "description" in rel_desc:
                            self.graph.add((rel_uri, self.db_ns.relationshipDescription, 
                                          Literal(rel_desc["description"])))
                        
                        if "cardinality" in rel_desc:
                            self.graph.add((rel_uri, self.db_ns.cardinality, 
                                          Literal(rel_desc["cardinality"])))
                        
                        if "business_rule" in rel_desc:
                            self.graph.add((rel_uri, self.db_ns.businessRule, 
                                          Literal(rel_desc["business_rule"])))
        
        logger.info("Finished applying semantic descriptions")
    
    def serialize_ontology(self) -> str:
        """Serialize the current ontology to Turtle format."""
        return self.graph.serialize(format="turtle")

    def get_enrichment_data(self, tables_info: List[TableInfo], sample_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate enrichment data structure for LLM processing.

        Args:
            tables_info: List of table information
            sample_data: Dictionary mapping table names to sample rows

        Returns:
            Dictionary with schema_data and instructions for LLM enrichment
        """
        schema_data = []

        for table_info in tables_info:
            table_data = {
                "table_name": table_info.name,
                "schema": table_info.schema,
                "row_count": table_info.row_count,
                "columns": []
            }

            # Add column information
            for column in table_info.columns:
                column_data = {
                    "name": column.name,
                    "data_type": column.data_type,
                    "is_nullable": column.is_nullable,
                    "is_primary_key": column.is_primary_key,
                    "is_foreign_key": column.is_foreign_key
                }
                if column.comment:
                    column_data["comment"] = column.comment
                if column.is_foreign_key and column.foreign_key_table:
                    column_data["foreign_key_table"] = column.foreign_key_table
                    column_data["foreign_key_column"] = column.foreign_key_column
                table_data["columns"].append(column_data)

            # Add sample data if available (limit to 3 rows)
            if table_info.name in sample_data and sample_data[table_info.name]:
                table_data["sample_data"] = sample_data[table_info.name][:3]

            schema_data.append(table_data)

        # Generate instructions for LLM enrichment
        instructions = {
            "task": "Enrich the database schema with semantic descriptions",
            "expected_format": {
                "classes": [
                    {
                        "original_name": "table_name",
                        "suggested_name": "SemanticName",
                        "description": "Business description"
                    }
                ],
                "properties": [
                    {
                        "table_name": "table_name",
                        "original_name": "column_name",
                        "suggested_name": "semanticPropertyName",
                        "description": "Business meaning"
                    }
                ],
                "relationships": [
                    {
                        "from_table": "source_table",
                        "to_table": "target_table",
                        "suggested_name": "semanticRelationshipName",
                        "description": "Relationship meaning"
                    }
                ]
            },
            "guidelines": [
                "Use clear, business-oriented terminology",
                "Provide meaningful descriptions based on table and column names and sample data",
                "Suggest appropriate semantic names that reflect business concepts",
                "For relationships, describe the business meaning of the association"
            ]
        }

        return {
            "schema_data": schema_data,
            "instructions": instructions
        }

    def apply_enrichment(self, enrichment_suggestions: Dict[str, List[Dict[str, Any]]]) -> None:
        """Apply enrichment suggestions to the ontology.

        Args:
            enrichment_suggestions: Dictionary containing enrichment suggestions for:
                - classes: Class-level enrichments (table names, descriptions)
                - properties: Property-level enrichments (column names, descriptions)
                - relationships: Relationship enrichments (foreign key descriptions)
        """
        logger.info("Applying enrichment suggestions to ontology")

        # Apply class enrichments
        if "classes" in enrichment_suggestions:
            for class_enrichment in enrichment_suggestions["classes"]:
                original_name = class_enrichment.get("original_name")
                suggested_name = class_enrichment.get("suggested_name")
                description = class_enrichment.get("description")

                if not original_name:
                    continue

                # Find the original class URI
                original_uri = self.base_uri[self._clean_name(original_name)]

                # Add suggested name as label if provided
                if suggested_name:
                    # Remove old label if exists
                    self.graph.remove((original_uri, RDFS.label, None))
                    self.graph.add((original_uri, RDFS.label, Literal(suggested_name)))

                # Add description as comment if provided
                if description:
                    # Remove old comment if exists
                    self.graph.remove((original_uri, RDFS.comment, None))
                    self.graph.add((original_uri, RDFS.comment, Literal(description)))

        # Apply property enrichments
        if "properties" in enrichment_suggestions:
            for prop_enrichment in enrichment_suggestions["properties"]:
                table_name = prop_enrichment.get("table_name")
                original_name = prop_enrichment.get("original_name")
                suggested_name = prop_enrichment.get("suggested_name")
                description = prop_enrichment.get("description")

                if not table_name or not original_name:
                    continue

                # Find the original property URI
                prop_name = f"{self._clean_name(table_name)}_{self._clean_name(original_name)}"
                prop_uri = self.base_uri[prop_name]

                # Add suggested name as label if provided
                if suggested_name:
                    self.graph.remove((prop_uri, RDFS.label, None))
                    self.graph.add((prop_uri, RDFS.label, Literal(suggested_name)))

                # Add description as comment if provided
                if description:
                    self.graph.remove((prop_uri, RDFS.comment, None))
                    self.graph.add((prop_uri, RDFS.comment, Literal(description)))

        # Apply relationship enrichments
        if "relationships" in enrichment_suggestions:
            for rel_enrichment in enrichment_suggestions["relationships"]:
                from_table = rel_enrichment.get("from_table")
                to_table = rel_enrichment.get("to_table")
                suggested_name = rel_enrichment.get("suggested_name")
                description = rel_enrichment.get("description")

                if not from_table or not to_table:
                    continue

                # Find the original relationship URI
                rel_name = f"{self._clean_name(from_table)}_has_{self._clean_name(to_table)}"
                rel_uri = self.base_uri[rel_name]

                # Add suggested name as label if provided
                if suggested_name:
                    self.graph.remove((rel_uri, RDFS.label, None))
                    self.graph.add((rel_uri, RDFS.label, Literal(suggested_name)))

                # Add description as comment if provided
                if description:
                    self.graph.remove((rel_uri, RDFS.comment, None))
                    self.graph.add((rel_uri, RDFS.comment, Literal(description)))

        logger.info("Finished applying enrichment suggestions")

    def enrich_with_llm(self, tables_info: List[TableInfo], sample_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate basic ontology with optional LLM enrichment.

        This is a placeholder method that generates a basic ontology.
        LLM enrichment is handled by the MCP tools layer.

        Args:
            tables_info: List of table information
            sample_data: Sample data for tables

        Returns:
            Serialized ontology in Turtle format
        """
        logger.info("Generating basic ontology (LLM enrichment handled by MCP tools)")
        return self.generate_from_schema(tables_info)



