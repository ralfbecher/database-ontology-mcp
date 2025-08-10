# Contributing to Database Ontology MCP Server

Thank you for your interest in contributing to the Database Ontology MCP Server! This document provides guidelines and information for contributors.

## 🎯 Project Overview

The Database Ontology MCP Server is an enhanced, production-ready MCP (Model Context Protocol) server that analyzes relational database schemas and generates high-quality RDF/OWL ontologies. The project emphasizes security, performance, reliability, and comprehensive LLM integration capabilities.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Access to PostgreSQL or Snowflake database for testing
- Basic understanding of:
  - Database schemas and relationships
  - RDF/OWL ontologies and semantic web concepts
  - MCP (Model Context Protocol)
  - Python development best practices

### Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/your-username/database-ontology-mcp.git
cd database-ontology-mcp
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies:**
```bash
pip install -r requirements.txt
# Install development tools
pip install -e ".[dev]"  # If using setup.py
# Or install dev dependencies from pyproject.toml
pip install pytest pytest-cov black isort flake8 mypy pre-commit
```

4. **Set up pre-commit hooks (recommended):**
```bash
pre-commit install
```

5. **Configure environment for testing:**
```bash
cp config/.env.template .env
# Edit .env with test database credentials
```

6. **Verify setup:**
```bash
python -m pytest tests/ -v
python run_server.py --help
```

## 🏗️ Project Architecture

### Key Components

- **`src/main.py`**: FastMCP server implementation with enhanced error handling
- **`src/database_manager.py`**: Database connectivity and schema analysis
- **`src/ontology_generator.py`**: RDF/OWL ontology generation and enrichment
- **`src/config.py`**: Configuration management and validation
- **`src/constants.py`**: Application constants and settings
- **`src/utils.py`**: Utility functions and helpers

### Design Principles

1. **Security First**: Input validation, credential protection, SQL injection prevention
2. **Performance**: Connection pooling, parallel processing, efficient resource usage
3. **Reliability**: Comprehensive error handling, graceful degradation, proper logging
4. **Maintainability**: Clear code structure, comprehensive tests, documentation
5. **Extensibility**: Modular design for easy feature additions and database support

## 📝 Contribution Guidelines

### Code Style

We use automated tools to maintain consistent code style:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

#### Standards
- **Line length**: 88 characters (Black default)
- **Import style**: isort with Black profile
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Required for all public classes and functions (Google style)
- **Naming**: snake_case for functions/variables, PascalCase for classes

### Testing Requirements

All contributions must include appropriate tests:

#### Test Coverage Requirements
- **New features**: Must have >90% test coverage
- **Bug fixes**: Must include regression tests
- **Refactoring**: Existing tests must continue to pass

#### Test Types
```bash
# Unit tests
python -m pytest tests/test_database_manager.py -v
python -m pytest tests/test_ontology_generator.py -v

# Integration tests
python -m pytest tests/test_server.py -v

# Coverage reporting
python -m pytest tests/ --cov=src --cov-report=html
```

#### Test Conventions
- Use descriptive test names: `test_connect_database_with_invalid_credentials_returns_error`
- Test both success and failure paths
- Mock external dependencies (databases, APIs)
- Use parameterized tests for multiple scenarios
- Include edge cases and boundary conditions

### Documentation Requirements

#### Code Documentation
```python
def analyze_table(self, table_name: str, schema_name: Optional[str] = None) -> Optional[TableInfo]:
    """Analyze a specific table and return detailed information.
    
    Args:
        table_name: Name of the table to analyze
        schema_name: Optional schema name, defaults to public schema
        
    Returns:
        TableInfo object with comprehensive table metadata, or None if table not found
        
    Raises:
        RuntimeError: If no database connection is established
        ValueError: If table_name contains invalid characters
        
    Example:
        >>> db_manager = DatabaseManager()
        >>> db_manager.connect_postgresql("localhost", 5432, "mydb", "user", "pass")
        >>> table_info = db_manager.analyze_table("users", "public")
        >>> print(f"Table has {len(table_info.columns)} columns")
    """
```

#### Update Documentation
- Update README.md for new features or API changes
- Add examples for new MCP tools
- Update environment variable documentation
- Include performance considerations for new features

### Commit Guidelines

We follow conventional commits for clear history:

```bash
# Feature additions
git commit -m "feat: add MySQL database support"
git commit -m "feat(ontology): implement RDFS subclass relationships"

# Bug fixes
git commit -m "fix: prevent SQL injection in table sampling"
git commit -m "fix(config): handle missing environment variables gracefully"

# Documentation
git commit -m "docs: add MySQL configuration examples"
git commit -m "docs(api): update MCP tool parameter descriptions"

# Tests
git commit -m "test: add comprehensive database manager tests"
git commit -m "test(integration): add end-to-end ontology generation tests"

# Refactoring
git commit -m "refactor: extract database connection logic"
git commit -m "refactor(utils): improve error handling utilities"
```

### Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass
   - Run code quality checks
   - Update documentation
   - Add changelog entry if applicable

2. **PR Description Template:**
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated
- [ ] No sensitive information committed
```

3. **Review Process:**
   - All PRs require at least one approving review
   - Address all review feedback
   - Ensure CI/CD pipeline passes
   - Squash commits if requested

## 🐛 Bug Reports

### Before Reporting
- Check existing issues to avoid duplicates
- Ensure you're using the latest version
- Test with minimal reproduction case

### Bug Report Template
```markdown
## Bug Description
Clear description of what the bug is.

## Environment
- OS: [e.g., Ubuntu 20.04, Windows 10]
- Python version: [e.g., 3.10.5]
- Server version: [e.g., v0.2.0]
- Database: [e.g., PostgreSQL 14.2, Snowflake]

## Reproduction Steps
1. Connect to database with...
2. Run MCP tool...
3. Observe error...

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Additional Context
- Error logs (with sensitive info redacted)
- Configuration details
- Screenshots if applicable
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
Clear description of the proposed feature.

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Solution
Detailed description of how you envision this feature working.

## Alternatives Considered
Other approaches you've considered.

## Additional Context
- Use cases
- Implementation ideas
- Related issues/PRs
```

## 🔧 Development Areas

### High Priority Areas
1. **Database Support**: Adding MySQL, SQLite, Oracle support
2. **Performance**: Query optimization, caching, streaming
3. **LLM Integration**: Enhanced enrichment algorithms, more LLM providers
4. **Security**: Advanced authentication, authorization, audit logging
5. **Monitoring**: Metrics, health checks, observability

### Medium Priority Areas
1. **UI/UX**: Web interface for ontology visualization
2. **Export Formats**: Additional RDF formats, documentation generation
3. **Schema Evolution**: Version tracking, change detection
4. **Testing**: Property-based testing, performance benchmarks
5. **Documentation**: Interactive examples, video tutorials

### Specialized Contributions Welcome
- **Database Experts**: Adding support for specialized databases
- **Semantic Web Experts**: Advanced ontology features, reasoning
- **Security Experts**: Penetration testing, security audits
- **Performance Engineers**: Optimization, scaling, benchmarking
- **UX Designers**: Interface design, user experience improvements

## 🎯 Specific Contribution Ideas

### Easy (Good First Issues)
- Add new SQL type mappings to XSD
- Improve error messages and user feedback
- Add configuration validation
- Enhance logging messages
- Write additional unit tests

### Medium
- Implement connection retry logic
- Add table sampling strategies
- Create ontology validation tools
- Implement caching layer
- Add performance metrics

### Advanced
- Add new database support (MySQL, Oracle, etc.)
- Implement streaming for large schemas
- Create advanced LLM integration
- Build web-based ontology editor
- Implement distributed processing

## 📚 Resources

### Learning Resources
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [RDF/OWL Primer](https://www.w3.org/TR/owl2-primer/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [FastMCP Framework](https://github.com/jlowin/fastmcp)

### Development Tools
- [Black Code Formatter](https://black.readthedocs.io/)
- [isort Import Sorter](https://isort.readthedocs.io/)
- [mypy Type Checker](https://mypy.readthedocs.io/)
- [pytest Testing Framework](https://pytest.org/)
- [pre-commit Hooks](https://pre-commit.com/)

## 🤝 Community

### Communication Channels
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code contributions, reviews

### Code of Conduct
We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please be respectful and inclusive in all interactions.

### Recognition
Contributors will be recognized in:
- README.md contributor section
- Release notes for significant contributions
- Special recognition for major features or improvements

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## ❓ Questions?

- Check existing documentation and issues first
- Create a GitHub Discussion for general questions
- Create a GitHub Issue for specific problems
- Reach out to maintainers for major architectural questions

Thank you for contributing to the Database Ontology MCP Server! 🚀