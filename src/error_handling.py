"""Centralized error handling and response management."""

import logging
import traceback
from enum import Enum
from typing import Dict, Any, Optional, Callable
from functools import wraps

from .security import audit_log_security_event, SecurityLevel

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Standardized error types for consistent handling."""
    VALIDATION = "validation_error"
    CONNECTION = "connection_error"
    RUNTIME = "runtime_error"
    INTERNAL = "internal_error"
    SECURITY = "security_error"
    PERMISSION = "permission_error"
    TIMEOUT = "timeout_error"
    NOT_FOUND = "not_found_error"


class DatabaseOntologyError(Exception):
    """Base exception class for database ontology MCP server."""
    
    def __init__(self, message: str, error_type: ErrorType = ErrorType.RUNTIME, 
                 details: Optional[str] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details
        self.original_error = original_error


class ValidationError(DatabaseOntologyError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorType.VALIDATION, details)


class ConnectionError(DatabaseOntologyError):
    """Exception for database connection errors."""
    
    def __init__(self, message: str, details: Optional[str] = None, original_error: Optional[Exception] = None):
        super().__init__(message, ErrorType.CONNECTION, details, original_error)


class SecurityError(DatabaseOntologyError):
    """Exception for security-related errors."""
    
    def __init__(self, message: str, details: Optional[str] = None, risk_level: SecurityLevel = SecurityLevel.HIGH):
        super().__init__(message, ErrorType.SECURITY, details)
        self.risk_level = risk_level


def create_error_response(
    message: str, 
    error_type: ErrorType, 
    details: Optional[str] = None,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """Create a standardized error response.
    
    Args:
        message: Human-readable error message
        error_type: Type of error (from ErrorType enum)
        details: Additional error details
        include_traceback: Whether to include stack trace (for debugging)
    
    Returns:
        Standardized error response dictionary
    """
    response = {
        "success": False,
        "error": message,
        "error_type": error_type.value,
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }
    
    if details:
        response["details"] = details
    
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    return response


def handle_database_errors(operation_name: str, include_traceback: bool = False):
    """Decorator for consistent database error handling.
    
    Args:
        operation_name: Name of the operation for logging
        include_traceback: Whether to include traceback in response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except SecurityError as e:
                logger.error(f"Security error in {operation_name}: {e.message}")
                audit_log_security_event(
                    f"security_error_{operation_name}",
                    {"error": e.message, "details": e.details},
                    e.risk_level
                )
                return create_error_response(
                    e.message, e.error_type, e.details, include_traceback
                )
            
            except ValidationError as e:
                logger.warning(f"Validation error in {operation_name}: {e.message}")
                return create_error_response(
                    e.message, e.error_type, e.details, include_traceback
                )
            
            except ConnectionError as e:
                logger.error(f"Connection error in {operation_name}: {e.message}")
                if e.original_error:
                    logger.debug(f"Original error: {e.original_error}")
                return create_error_response(
                    e.message, e.error_type, e.details, include_traceback
                )
            
            except DatabaseOntologyError as e:
                logger.error(f"Application error in {operation_name}: {e.message}")
                return create_error_response(
                    e.message, e.error_type, e.details, include_traceback
                )
            
            except ImportError as e:
                logger.error(f"Missing dependency in {operation_name}: {e}")
                return create_error_response(
                    f"Missing required dependency: {str(e)}",
                    ErrorType.RUNTIME,
                    "Please install missing dependencies",
                    include_traceback
                )
            
            except PermissionError as e:
                logger.error(f"Permission error in {operation_name}: {e}")
                return create_error_response(
                    "Insufficient permissions for operation",
                    ErrorType.PERMISSION,
                    str(e),
                    include_traceback
                )
            
            except TimeoutError as e:
                logger.error(f"Timeout in {operation_name}: {e}")
                return create_error_response(
                    "Operation timed out",
                    ErrorType.TIMEOUT,
                    str(e),
                    include_traceback
                )
            
            except Exception as e:
                logger.error(f"Unexpected error in {operation_name}: {type(e).__name__}: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                return create_error_response(
                    f"Internal server error: {str(e)}",
                    ErrorType.INTERNAL,
                    f"Unexpected {type(e).__name__}",
                    include_traceback
                )
        
        return wrapper
    return decorator


def handle_mcp_tool_errors(tool_name: str):
    """Specialized decorator for MCP tool error handling.
    
    MCP tools need specific error response formats.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Ensure result is JSON serializable
                if hasattr(result, 'model_dump'):  # Pydantic model
                    return result.model_dump()
                elif hasattr(result, '__dict__'):  # Dataclass or object
                    return result.__dict__
                else:
                    return result
                
            except Exception as e:
                logger.error(f"Error in MCP tool {tool_name}: {type(e).__name__}: {e}")
                
                # For MCP tools, we return the error response directly
                # (not wrapped in another structure)
                error_response = create_error_response(
                    str(e),
                    ErrorType.RUNTIME,
                    f"Error in {tool_name}",
                    include_traceback=False  # Don't expose internal details to MCP clients
                )
                
                return error_response
        
        return wrapper
    return decorator


class ErrorContext:
    """Context manager for error handling with cleanup."""
    
    def __init__(self, operation_name: str, cleanup_func: Optional[Callable] = None):
        self.operation_name = operation_name
        self.cleanup_func = cleanup_func
        self.start_time = None
    
    def __enter__(self):
        self.start_time = __import__("time").time()
        logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = __import__("time").time() - self.start_time
        
        if exc_type is None:
            logger.debug(f"Operation completed successfully: {self.operation_name} ({duration:.2f}s)")
        else:
            logger.error(f"Operation failed: {self.operation_name} ({duration:.2f}s) - {exc_type.__name__}: {exc_val}")
            
            # Run cleanup if provided
            if self.cleanup_func:
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed for {self.operation_name}: {cleanup_error}")
        
        # Don't suppress exceptions
        return False


def validate_required_params(params: Dict[str, Any], required_keys: list) -> None:
    """Validate that required parameters are present and not None.
    
    Args:
        params: Dictionary of parameters
        required_keys: List of required parameter names
    
    Raises:
        ValidationError: If any required parameters are missing
    """
    missing_keys = []
    for key in required_keys:
        if key not in params or params[key] is None:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValidationError(
            f"Missing required parameters: {', '.join(missing_keys)}",
            details=f"Required parameters: {', '.join(required_keys)}"
        )


def validate_parameter_types(params: Dict[str, Any], type_specs: Dict[str, type]) -> None:
    """Validate parameter types.
    
    Args:
        params: Dictionary of parameters
        type_specs: Dictionary mapping parameter names to expected types
    
    Raises:
        ValidationError: If any parameters have wrong types
    """
    type_errors = []
    for key, expected_type in type_specs.items():
        if key in params and params[key] is not None:
            if not isinstance(params[key], expected_type):
                type_errors.append(f"{key} should be {expected_type.__name__}, got {type(params[key]).__name__}")
    
    if type_errors:
        raise ValidationError(
            f"Parameter type errors: {'; '.join(type_errors)}",
            details="Check parameter types in your request"
        )


# Convenience functions for common error scenarios
def connection_required_error() -> Dict[str, Any]:
    """Standard error for when database connection is required but not established."""
    return create_error_response(
        "No database connection established",
        ErrorType.CONNECTION,
        "Use connect_database tool to establish a connection first"
    )


def invalid_identifier_error(identifier: str) -> Dict[str, Any]:
    """Standard error for invalid database identifiers."""
    return create_error_response(
        f"Invalid identifier: {identifier}",
        ErrorType.VALIDATION,
        "Identifiers must contain only letters, numbers, underscores, and hyphens"
    )


def table_not_found_error(table_name: str, schema_name: Optional[str] = None) -> Dict[str, Any]:
    """Standard error for when a table is not found."""
    full_name = f"{schema_name}.{table_name}" if schema_name else table_name
    return create_error_response(
        f"Table not found: {full_name}",
        ErrorType.NOT_FOUND,
        "Check that the table name and schema are correct"
    )
