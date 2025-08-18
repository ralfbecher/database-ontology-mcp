"""Utility functions for the Database Ontology MCP Server."""

import json
import logging
import logging.config
import re
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Union


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_obj.update(record.extra)
        
        return json.dumps(log_obj)


def setup_logging(log_level: str = "INFO", structured: bool = True) -> logging.Logger:
    """Setup logging configuration with optional structured logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        structured: Whether to use structured (JSON) logging
    
    Returns:
        Configured logger instance
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create module-specific logger
    logger = logging.getLogger("database_ontology_mcp")
    logger.info(f"Logging configured at {log_level} level")
    
    return logger


def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize sensitive information for logging.
    
    Args:
        data: Dictionary potentially containing sensitive information
    
    Returns:
        Sanitized dictionary safe for logging
    """
    sensitive_keys = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'credential',
        'auth', 'authorization', 'api_key', 'access_key', 'private_key'
    }
    
    sanitized = {}
    for key, value in data.items():
        lower_key = key.lower()
        if any(sensitive in lower_key for sensitive in sensitive_keys):
            if value:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = None
        else:
            # Recursively sanitize nested dictionaries
            if isinstance(value, dict):
                sanitized[key] = sanitize_for_logging(value)
            else:
                sanitized[key] = value
    
    return sanitized


def sanitize_sql_for_logging(sql_query: str, max_length: int = 200) -> str:
    """Safely sanitize SQL query for logging without exposing credentials.
    
    Args:
        sql_query: SQL query to sanitize
        max_length: Maximum length of logged query (default: 200 chars)
    
    Returns:
        Sanitized SQL query safe for logging
    """
    if not sql_query:
        return ""
    
    # Remove potential passwords, secrets, or sensitive data from SQL
    # This is a basic approach - more sophisticated parsing could be added
    sanitized = sql_query.strip()
    
    # Remove common credential patterns (case insensitive)
    credential_patterns = [
        r"password\s*[=:]\s*['\"][^'\"]*['\"]",
        r"pwd\s*[=:]\s*['\"][^'\"]*['\"]", 
        r"secret\s*[=:]\s*['\"][^'\"]*['\"]",
        r"token\s*[=:]\s*['\"][^'\"]*['\"]",
        r"key\s*[=:]\s*['\"][^'\"]*['\"]",
    ]
    
    for pattern in credential_patterns:
        sanitized = re.sub(pattern, "[CREDENTIAL_REDACTED]", sanitized, flags=re.IGNORECASE)
    
    # Truncate if too long and add ellipsis
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    # Replace multiple whitespaces with single spaces for cleaner logging
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    return sanitized


def validate_uri(uri: str) -> bool:
    """Validate URI format.
    
    Args:
        uri: URI string to validate
    
    Returns:
        True if valid URI format, False otherwise
    """
    if not uri:
        return False
    
    # Basic URI validation regex
    uri_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return uri_pattern.match(uri) is not None


def format_bytes(bytes_value: int) -> str:
    """Format bytes value in human-readable format.
    
    Args:
        bytes_value: Number of bytes
    
    Returns:
        Human-readable string (e.g., "1.5 MB")
    """
    if bytes_value == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while bytes_value >= 1024.0 and i < len(size_names) - 1:
        bytes_value /= 1024.0
        i += 1
    
    return f"{bytes_value:.1f} {size_names[i]}"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length with optional suffix.
    
    Args:
        text: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating
    
    Returns:
        Truncated string
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize object to JSON, handling non-serializable types.
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON string representation
    """
    def json_handler(obj):
        """Handle non-serializable objects."""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # objects with attributes
            return str(obj)
        elif isinstance(obj, bytes):
            return obj.hex()
        else:
            return str(obj)
    
    try:
        return json.dumps(obj, default=json_handler, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"})


def calculate_table_stats(tables_info: list) -> Dict[str, Any]:
    """Calculate statistics for a list of table information.
    
    Args:
        tables_info: List of TableInfo objects
    
    Returns:
        Dictionary containing table statistics
    """
    if not tables_info:
        return {
            "total_tables": 0,
            "total_columns": 0,
            "total_rows": 0,
            "tables_with_fks": 0,
            "avg_columns_per_table": 0
        }
    
    total_tables = len(tables_info)
    total_columns = sum(len(table.columns) for table in tables_info)
    total_rows = sum(table.row_count or 0 for table in tables_info)
    tables_with_fks = sum(1 for table in tables_info if table.foreign_keys)
    
    return {
        "total_tables": total_tables,
        "total_columns": total_columns,
        "total_rows": total_rows,
        "tables_with_fks": tables_with_fks,
        "avg_columns_per_table": round(total_columns / total_tables, 2) if total_tables > 0 else 0,
        "fk_percentage": round((tables_with_fks / total_tables) * 100, 1) if total_tables > 0 else 0
    }


class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(f"{self.operation_name} failed after {duration:.2f}s: {exc_val}")
        else:
            self.logger.debug(f"Completed {self.operation_name} in {duration:.2f}s")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get duration in seconds if timing is complete."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be doubled for each retry)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logging.getLogger(__name__).warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logging.getLogger(__name__).error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator