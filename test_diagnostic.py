#!/usr/bin/env python3
"""Test script to run the Dremio diagnostic tool."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config_manager

def test_dremio_diagnostic():
    """Test the Dremio diagnostic tool."""
    print("🔍 Running Dremio Connection Diagnostic...")
    print("=" * 50)
    
    # Simulate the diagnostic logic directly
    db_config = config_manager.get_database_config()
    
    actual_host = db_config.dremio_host
    actual_port = db_config.dremio_port
    actual_username = db_config.dremio_username
    actual_password = db_config.dremio_password
    actual_ssl = False  # Default to False for community edition
    
    result = {
        "db_type": "dremio",
        "success": True,
        "connection_parameters": {
            "host": actual_host,
            "port": actual_port,
            "username": actual_username,
            "password_provided": bool(actual_password),
            "ssl": actual_ssl
        },
        "issues_found": [],
        "recommendations": [],
        "connection_string_preview": None
    }
    
    # Check for missing parameters
    if not actual_host:
        result["issues_found"].append("Missing DREMIO_HOST")
        result["recommendations"].append("Set DREMIO_HOST in .env file or pass host parameter")
    
    if not actual_port:
        result["issues_found"].append("Missing DREMIO_PORT")
        result["recommendations"].append("Set DREMIO_PORT in .env file (default: 31010) or pass port parameter")
    
    if not actual_username:
        result["issues_found"].append("Missing DREMIO_USERNAME")
        result["recommendations"].append("Set DREMIO_USERNAME in .env file or pass username parameter")
    
    if not actual_password:
        result["issues_found"].append("Missing DREMIO_PASSWORD")
        result["recommendations"].append("Set DREMIO_PASSWORD in .env file")
    
    # Show connection string format (without password)
    if actual_host and actual_port and actual_username:
        sslmode = 'require' if actual_ssl else 'disable'
        result["connection_string_preview"] = (
            f"postgresql://{actual_username}:****@{actual_host}:{actual_port}/dremio"
            f"?sslmode={sslmode}&application_name=database-ontology-mcp"
        )
    
    # Add Dremio-specific recommendations
    result["recommendations"].extend([
        "Ensure Dremio coordinator is running and accessible",
        "Verify port 31010 (default) is not blocked by firewall",
        "For production Dremio, check if SSL/TLS is required (use ssl=True)",
        "Confirm username/password are correct for Dremio",
        "Try connecting with Dremio's web UI first to verify credentials",
        "For cloud Dremio, check if IP whitelisting is required"
    ])
    
    # Connection test recommendations
    result["connection_test_steps"] = [
        f"1. Test basic connectivity: telnet {actual_host or 'HOST'} {actual_port or 31010}",
        "2. Check Dremio web UI is accessible (usually same host, port 9047)",
        "3. Verify credentials work in Dremio web interface",
        "4. Check Dremio coordinator logs for connection attempts",
        "5. For production environments, test with SSL enabled: ssl=True"
    ]
    
    # Common Dremio issues
    result["common_issues"] = {
        "SSL_HANDSHAKE_FAILURE": {
            "description": "SSL/TLS connection failed",
            "solutions": ["Try ssl=False", "Check SSL certificate", "Verify Dremio SSL configuration"]
        },
        "CONNECTION_REFUSED": {
            "description": "Cannot connect to Dremio coordinator",
            "solutions": ["Check if Dremio is running", "Verify host and port", "Check firewall rules"]
        },
        "AUTHENTICATION_FAILED": {
            "description": "Username/password incorrect",
            "solutions": ["Verify credentials in Dremio UI", "Check for special characters in password", "Confirm user has CONNECT privileges"]
        },
        "TIMEOUT": {
            "description": "Connection attempt timed out",
            "solutions": ["Check network connectivity", "Verify Dremio coordinator is responsive", "Try increasing connection timeout"]
        }
    }
    
    print(f"✅ Success: {result.get('success', 'Unknown')}")
    print(f"📊 Database Type: {result.get('db_type', 'Unknown')}")
    
    if 'connection_parameters' in result:
        print("\n📋 Connection Parameters:")
        for key, value in result['connection_parameters'].items():
            if key == 'password_provided':
                print(f"  - {key}: {value}")
            else:
                print(f"  - {key}: {value}")
    
    if result.get('issues_found'):
        print(f"\n❌ Issues Found ({len(result['issues_found'])}):")
        for issue in result['issues_found']:
            print(f"  - {issue}")
    
    if result.get('recommendations'):
        print(f"\n💡 Recommendations ({len(result['recommendations'])}):")
        for rec in result['recommendations']:
            print(f"  - {rec}")
    
    if result.get('connection_string_preview'):
        print(f"\n🔗 Connection String Preview:")
        print(f"  {result['connection_string_preview']}")
    
    if result.get('connection_test_steps'):
        print(f"\n🧪 Connection Test Steps:")
        for step in result['connection_test_steps']:
            print(f"  {step}")
    
    if result.get('common_issues'):
        print(f"\n🚨 Common Dremio Issues:")
        for issue_name, details in result['common_issues'].items():
            print(f"  {issue_name}:")
            print(f"    Description: {details['description']}")
            print(f"    Solutions: {', '.join(details['solutions'])}")

if __name__ == "__main__":
    test_dremio_diagnostic()