"""Configuration management for the Database Ontology MCP Server."""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from .constants import DEFAULT_BASE_URI, DEFAULT_POSTGRES_PORT, DEFAULT_SNOWFLAKE_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration settings."""
    log_level: str
    ontology_base_uri: str
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.ontology_base_uri.endswith('/'):
            self.ontology_base_uri += '/'


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    # PostgreSQL settings
    postgres_host: Optional[str] = None
    postgres_port: int = DEFAULT_POSTGRES_PORT
    postgres_database: Optional[str] = None
    postgres_username: Optional[str] = None
    postgres_password: Optional[str] = None
    
    # Snowflake settings
    snowflake_account: Optional[str] = None
    snowflake_username: Optional[str] = None
    snowflake_password: Optional[str] = None
    snowflake_warehouse: Optional[str] = None
    snowflake_database: Optional[str] = None
    snowflake_schema: str = DEFAULT_SNOWFLAKE_SCHEMA


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self):
        """Initialize configuration manager."""
        load_dotenv()
        self._server_config: Optional[ServerConfig] = None
        self._db_config: Optional[DatabaseConfig] = None
    
    def get_server_config(self) -> ServerConfig:
        """Get server configuration."""
        if self._server_config is None:
            self._server_config = ServerConfig(
                log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
                ontology_base_uri=os.getenv("ONTOLOGY_BASE_URI", DEFAULT_BASE_URI)
            )
            logger.info("Server configuration loaded")
        return self._server_config
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        if self._db_config is None:
            self._db_config = DatabaseConfig(
                postgres_host=os.getenv("POSTGRES_HOST"),
                postgres_port=int(os.getenv("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
                postgres_database=os.getenv("POSTGRES_DATABASE"),
                postgres_username=os.getenv("POSTGRES_USERNAME"),
                postgres_password=os.getenv("POSTGRES_PASSWORD"),
                snowflake_account=os.getenv("SNOWFLAKE_ACCOUNT"),
                snowflake_username=os.getenv("SNOWFLAKE_USERNAME"),
                snowflake_password=os.getenv("SNOWFLAKE_PASSWORD"),
                snowflake_warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
                snowflake_database=os.getenv("SNOWFLAKE_DATABASE"),
                snowflake_schema=os.getenv("SNOWFLAKE_SCHEMA", DEFAULT_SNOWFLAKE_SCHEMA)
            )
            logger.info("Database configuration loaded")
        return self._db_config
    
    def validate_db_config(self, db_type: str) -> Dict[str, Any]:
        """Validate database configuration for a specific type."""
        config = self.get_database_config()
        missing_params = []
        
        if db_type == "postgresql":
            required_fields = {
                "host": config.postgres_host,
                "port": config.postgres_port,
                "database": config.postgres_database,
                "username": config.postgres_username,
                "password": config.postgres_password
            }
        elif db_type == "snowflake":
            required_fields = {
                "account": config.snowflake_account,
                "username": config.snowflake_username,
                "password": config.snowflake_password,
                "warehouse": config.snowflake_warehouse,
                "database": config.snowflake_database
            }
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        missing_params = [k for k, v in required_fields.items() if not v]
        
        return {
            "valid": len(missing_params) == 0,
            "missing_params": missing_params,
            "config": required_fields
        }


# Global configuration manager instance
config_manager = ConfigManager()