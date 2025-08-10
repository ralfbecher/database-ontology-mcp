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

class OntologyGenerator:
    """Generates an ontology from a database schema."""

    def __init__(self, base_uri: str = DEFAULT_BASE_URI):
        self.graph = Graph()
        self.base_uri = Namespace(base_uri)
        self.graph.bind("ns", self.base_uri)
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
        """Add a single table and its columns to the ontology."""
        # Create proper URI for table class
        table_uri = self.base_uri[self._clean_name(table_info.name)]

        # Define table as a class
        self.graph.add((table_uri, RDF.type, OWL.Class))
        self.graph.add((table_uri, RDFS.label, Literal(table_info.name)))
        if table_info.comment:
            self.graph.add((table_uri, RDFS.comment, Literal(table_info.comment)))

        # Add columns as properties
        for column in table_info.columns:
            self._add_column_to_ontology(table_uri, column, table_info.name)

        # Define relationships
        for fk in table_info.foreign_keys:
            self._add_relationship_to_ontology(table_uri, fk, table_info.name)

    def _add_column_to_ontology(self, table_uri: URIRef, column: ColumnInfo, table_name: str):
        """Add a column as a data property to the ontology."""
        # Create proper property URI
        prop_name = f"{self._clean_name(table_name)}_{self._clean_name(column.name)}"
        prop_uri = self.base_uri[prop_name]

        # Choose property type based on whether it's a foreign key
        if column.is_foreign_key:
            # Foreign key columns are object properties (handled in relationships)
            return
        else:
            # Regular data property
            self.graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        
        self.graph.add((prop_uri, RDFS.domain, table_uri))
        self.graph.add((prop_uri, RDFS.label, Literal(column.name)))
        
        # Map SQL data types to proper XSD types
        xsd_type = self._map_sql_to_xsd(column.data_type)
        if xsd_type:
            self.graph.add((prop_uri, RDFS.range, xsd_type))
        
        # Add cardinality constraints for primary keys and nullable fields
        if column.is_primary_key:
            # Primary keys are required and unique
            restriction = self.base_uri[f"{prop_name}_PK_Restriction"]
            self.graph.add((restriction, RDF.type, OWL.Restriction))
            self.graph.add((restriction, OWL.onProperty, prop_uri))
            self.graph.add((restriction, OWL.cardinality, Literal(1)))
            self.graph.add((table_uri, RDFS.subClassOf, restriction))
        elif not column.is_nullable:
            # Required fields have minimum cardinality 1
            restriction = self.base_uri[f"{prop_name}_Required_Restriction"]
            self.graph.add((restriction, RDF.type, OWL.Restriction))
            self.graph.add((restriction, OWL.onProperty, prop_uri))
            self.graph.add((restriction, OWL.minCardinality, Literal(1)))
            self.graph.add((table_uri, RDFS.subClassOf, restriction))
        
        if column.comment:
            self.graph.add((prop_uri, RDFS.comment, Literal(column.comment)))

    def _add_relationship_to_ontology(self, table_uri: URIRef, fk: Dict[str, str], table_name: str):
        """Add a foreign key relationship as an object property."""
        # Create descriptive relationship name
        rel_name = f"{self._clean_name(table_name)}_has_{self._clean_name(fk['referenced_table'])}"
        prop_uri = self.base_uri[rel_name]
        referenced_table_uri = self.base_uri[self._clean_name(fk['referenced_table'])]

        self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
        self.graph.add((prop_uri, RDFS.domain, table_uri))
        self.graph.add((prop_uri, RDFS.range, referenced_table_uri))
        self.graph.add((prop_uri, RDFS.label, Literal(f"{table_name} has {fk['referenced_table']}")))
        
        # Add inverse relationship
        inverse_rel_name = f"{self._clean_name(fk['referenced_table'])}_referenced_by_{self._clean_name(table_name)}"
        inverse_prop_uri = self.base_uri[inverse_rel_name]
        self.graph.add((inverse_prop_uri, RDF.type, OWL.ObjectProperty))
        self.graph.add((inverse_prop_uri, RDFS.domain, referenced_table_uri))
        self.graph.add((inverse_prop_uri, RDFS.range, table_uri))
        self.graph.add((inverse_prop_uri, RDFS.label, Literal(f"{fk['referenced_table']} referenced by {table_name}")))
        
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
        
        # Integer types
        if any(t in sql_type for t in ["int", "serial", "bigint", "smallint"]):
            return XSD.integer
        if "tinyint" in sql_type:
            return XSD.byte
        
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

    def enrich_with_llm(self, schema_info: List[TableInfo], data_samples: Dict[str, List[Dict[str, Any]]]) -> str:
        """Enriches the ontology with LLM insights via MCP tools."""
        logger.info("LLM enrichment will be handled by MCP prompts and tools.")
        logger.info("Use the 'get_enrichment_data' and 'apply_enrichment' MCP tools for enrichment.")
        
        # For now, return the basic ontology
        # The enrichment will be handled through MCP tools
        return self.graph.serialize(format="turtle")

    def get_enrichment_data(self, schema_info: List[TableInfo], data_samples: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Prepares structured data for LLM enrichment analysis."""
        schema_summary = []
        for table in schema_info:
            columns = []
            for col in table.columns:
                col_info = {
                    "name": col.name,
                    "data_type": col.data_type,
                    "is_primary_key": col.is_primary_key,
                    "is_foreign_key": col.is_foreign_key,
                    "is_nullable": col.is_nullable,
                    "comment": col.comment
                }
                if col.is_foreign_key:
                    col_info["foreign_key_table"] = col.foreign_key_table
                    col_info["foreign_key_column"] = col.foreign_key_column
                columns.append(col_info)
            
            table_dict = {
                "table_name": table.name,
                "schema": table.schema,
                "columns": columns,
                "foreign_keys": table.foreign_keys,
                "row_count": table.row_count,
                "comment": table.comment
            }
            if data_samples.get(table.name):
                # Limit sample data to avoid token limit issues
                table_dict["sample_data"] = data_samples[table.name][:3]
            schema_summary.append(table_dict)

        return {
            "schema_data": schema_summary,
            "instructions": {
                "task": "Enrich a basic RDF ontology generated from a relational database schema",
                "expected_format": {
                    "classes": [
                        {
                            "original_name": "<original_table_name>",
                            "suggested_name": "<SuggestedClassNameInPascalCase>",
                            "description": "<A detailed description of the class's purpose>"
                        }
                    ],
                    "properties": [
                        {
                            "table_name": "<table_name>",
                            "original_name": "<original_column_name>",
                            "suggested_name": "<suggestedPropertyNameInCamelCase>",
                            "description": "<A detailed description of the property>"
                        }
                    ],
                    "relationships": [
                        {
                            "from_table": "<table_with_fk>",
                            "to_table": "<referenced_table>",
                            "suggested_name": "<suggestedRelationshipNameInCamelCase>",
                            "description": "<A detailed description of what the relationship represents>"
                        }
                    ]
                },
                "guidelines": [
                    "Class names should be in PascalCase",
                    "Property and relationship names should be in camelCase",
                    "Descriptions should be clear, concise, and explain the semantic meaning",
                    "If you have no suggestion for a name, use the original name but still provide a description",
                    "Only include elements for which you can provide a meaningful description or a better name",
                    "Focus on the most important tables and relationships first"
                ]
            }
        }

    def apply_enrichment(self, enrichment_data: Dict[str, Any]):
        """Applies the LLM's suggestions to the RDF graph."""
        logger.info("Applying LLM enrichment to the ontology graph...")

        # Update class labels and comments
        for class_sugg in enrichment_data.get("classes", []):
            original_uri = self.base_uri[self._clean_name(class_sugg['original_name'])]
            if (original_uri, RDF.type, OWL.Class) in self.graph:
                new_label = class_sugg.get('suggested_name', class_sugg['original_name'])
                self.graph.set((original_uri, RDFS.label, Literal(new_label)))
                if class_sugg.get('description'):
                    self.graph.add((original_uri, RDFS.comment, Literal(class_sugg['description'])))

        # Update property labels and comments
        for prop_sugg in enrichment_data.get("properties", []):
            table_name = prop_sugg['table_name']
            original_prop_name = f"{self._clean_name(table_name)}_{self._clean_name(prop_sugg['original_name'])}"
            prop_uri = self.base_uri[original_prop_name]
            if (prop_uri, RDF.type, OWL.DatatypeProperty) in self.graph:
                new_label = prop_sugg.get('suggested_name', prop_sugg['original_name'])
                self.graph.set((prop_uri, RDFS.label, Literal(new_label)))
                if prop_sugg.get('description'):
                    self.graph.add((prop_uri, RDFS.comment, Literal(prop_sugg['description'])))

        # Update relationship labels and comments
        for rel_sugg in enrichment_data.get("relationships", []):
            from_table = self._clean_name(rel_sugg['from_table'])
            to_table = self._clean_name(rel_sugg['to_table'])
            original_rel_name = f"{from_table}_has_{to_table}"
            rel_uri = self.base_uri[original_rel_name]
            if (rel_uri, RDF.type, OWL.ObjectProperty) in self.graph:
                new_label = rel_sugg.get('suggested_name', original_rel_name)
                self.graph.set((rel_uri, RDFS.label, Literal(new_label)))
                if rel_sugg.get('description'):
                    self.graph.add((rel_uri, RDFS.comment, Literal(rel_sugg['description'])))
        
        logger.info("Finished applying LLM enrichment.")

    def serialize_ontology(self) -> str:
        """Serialize the current ontology to Turtle format."""
        return self.graph.serialize(format="turtle")

