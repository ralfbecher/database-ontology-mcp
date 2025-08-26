"""SQL query validation and execution tools."""

import logging
from typing import Dict, Any

from ..database_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Global database manager instance
_db_manager: DatabaseManager = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def create_error_response(message: str, error_type: str, details: str = None) -> Dict[str, Any]:
    """Create a standardized error response."""
    error_response = {
        "success": False,
        "error": message,
        "error_type": error_type
    }
    if details:
        error_response["details"] = details
    return error_response


def validate_sql_syntax(sql_query: str) -> Dict[str, Any]:
    """Validate SQL query syntax using database-level validation.
    
    Uses the database's own SQL parser to provide accurate syntax validation
    and meaningful error messages for query correction.
    
    Args:
        sql_query: SQL query to validate
        
    Returns:
        Dictionary with validation results including any errors and suggestions
    """
    try:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        validation_result = db_manager.validate_sql_syntax(sql_query)
        logger.info(f"SQL validation completed: {'valid' if validation_result['is_valid'] else 'invalid'}")
        return validation_result
        
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"SQL validation error: {str(e)}",
            "error_type": "validation_error"
        }


def execute_sql_query(
    sql_query: str, 
    limit: int = 1000
) -> Dict[str, Any]:
    """Execute a validated SQL query and return results safely.
    
    ## 🚀 **STREAMLINED WORKFLOW** (Only 4 Steps):

    1. **Connect**: Use `connect_database()` to establish connection
    2. **Analyze**: Use `get_analysis_context()` - gets schema + ontology + relationships automatically
    3. **Validate**: Use `validate_sql_syntax()` before execution  
    4. **Execute**: Use `execute_sql_query()` to run validated queries

    ## 🎯 **Using the Ontology for Accurate SQL**:
    The `get_analysis_context()` tool provides an ontology containing:
    - **Ready-to-use SQL column references**: `customers.customer_id`, `orders.order_total`
    - **Complete JOIN conditions**: `orders.customer_id = customers.customer_id`
    - **Business context**: "Customer information and profile data"
    
    Extract these from the ontology TTL format and use them directly in your SQL queries.

    ## 🚨 CRITICAL SQL TRAP PREVENTION PROTOCOL 🚨

    ### MANDATORY PRE-EXECUTION CHECKLIST

    **1. 🔍 RELATIONSHIP ANALYSIS (REQUIRED)**
    - ALWAYS call `get_table_relationships()` first
    - Identify ALL 1:many relationships in your query
    - Flag any table appearing on "many" side of multiple relationships

    **2. 🎯 FAN-TRAP DETECTION (CRITICAL)**

    **IMMEDIATE RED FLAGS:**
    - ❌ Sales + Shipments + SUM() = GUARANTEED FAN-TRAP
    - ❌ Any fact table + dimension + aggregation = HIGH RISK
    - ❌ Multiple LEFT JOINs + GROUP BY = DANGER ZONE
    - ❌ Joining 3+ tables with SUM/COUNT/AVG = LIKELY INFLATED RESULTS

    **PATTERN CHECK:**
    ```
    If query has: FROM tableA JOIN tableB JOIN tableC 
    WHERE tableA→tableB (1:many) AND tableA→tableC (1:many)
    Then: GUARANTEED CARTESIAN PRODUCT MULTIPLICATION
    Result: SUM(tableA.amount) will be artificially inflated!
    ```

    **3. 🛯f MANDATORY VALIDATION**
    - Call `validate_sql_syntax()` before execution
    - Review warnings about query complexity
    - Check for multiple table joins with aggregation

    ## ✅ SAFE QUERY PATTERNS

    ### 🔒 PATTERN 1 - UNION APPROACH (RECOMMENDED FOR MULTI-FACT)

    **Best for:** Multiple fact tables (sales, shipments, returns, etc.)

    ```sql
    WITH unified_facts AS (
        -- Sales facts
        SELECT 
            client_id, 
            product_id, 
            sales_amount as amount, 
            0 as shipment_qty, 
            0 as return_qty,
            'SALES' as fact_type
        FROM sales
        
        UNION ALL
        
        -- Shipment facts  
        SELECT 
            client_id, 
            product_id, 
            0 as amount, 
            shipment_quantity, 
            0 as return_qty,
            'SHIPMENT' as fact_type
        FROM shipments s JOIN sales sal ON s.sales_id = sal.id
        
        UNION ALL
        
        -- Return facts
        SELECT 
            client_id, 
            product_id, 
            0 as amount, 
            0 as shipment_qty, 
            return_quantity,
            'RETURN' as fact_type  
        FROM returns r JOIN sales sal ON r.sales_id = sal.id
    )
    SELECT 
        client_id,
        product_id,
        SUM(amount) as total_sales,
        SUM(shipment_qty) as total_shipped,
        SUM(return_qty) as total_returned
    FROM unified_facts 
    GROUP BY client_id, product_id;
    ```

    **Advantages:**
    - ✅ Natural fan-trap immunity by design
    - ✅ Unified data model for consistent aggregation
    - ✅ Easy to extend with additional fact types
    - ✅ Single aggregation logic for all measures
    - ✅ Better performance with fewer table scans

    ### 🔒 PATTERN 2 - SEPARATE AGGREGATION (LEGACY APPROACH)

    **Use when:** UNION approach is not suitable

    ```sql
    WITH fact1_totals AS (
        SELECT key, SUM(amount) as total_amount 
        FROM fact1 GROUP BY key
    ),
    fact2_totals AS (
        SELECT key, SUM(quantity) as total_quantity 
        FROM fact2 GROUP BY key
    )
    SELECT 
        f1.key, 
        f1.total_amount,
        COALESCE(f2.total_quantity, 0) as total_quantity
    FROM fact1_totals f1 
    LEFT JOIN fact2_totals f2 ON f1.key = f2.key;
    ```

    ### 🔒 PATTERN 3 - DISTINCT AGGREGATION (USE CAREFULLY)

    **Warning:** Only use when you fully understand the data relationships

    ```sql
    SELECT 
        key, 
        SUM(DISTINCT fact1.amount) as total_amount,
        SUM(fact2.quantity) as total_quantity 
    FROM fact1 
    LEFT JOIN fact2 ON fact1.id = fact2.fact1_id 
    GROUP BY key;
    ```

    ### 🔒 PATTERN 4 - WINDOW FUNCTIONS

    **For:** Complex analytical queries with preserved granularity

    ```sql
    SELECT DISTINCT 
        key, 
        SUM(amount) OVER (PARTITION BY key) as total_amount,
        pre_aggregated_quantity
    FROM fact1 
    LEFT JOIN (
        SELECT key, SUM(qty) as pre_aggregated_quantity 
        FROM fact2 GROUP BY key
    ) f2 USING(key);
    ```

    ## 🔄 RESULT VALIDATION (POST-EXECUTION)

    **Always verify results make business sense:**
    - Compare totals with business expectations
    - Verify: `SELECT SUM(amount) FROM base_table` vs your query result
    - Check row counts are reasonable
    - If results seem too high → likely fan-trap occurred

    ## 📝 COMMON DEADLY COMBINATIONS TO AVOID

    ❌ **Never do these without proper fan-trap prevention:**
    - `sales LEFT JOIN shipments + SUM(sales.amount)`
    - `orders LEFT JOIN order_items LEFT JOIN products + SUM(orders.total)`
    - `customers LEFT JOIN transactions LEFT JOIN transaction_items + aggregation`
    - Any query joining parent→child1 + parent→child2 with SUM/COUNT

    ## 🎯 RELATIONSHIP ANALYSIS EXAMPLES

    **SAFE (1:1 relationships):**
    ```
    customers → customer_profiles (1:1) ✅
    ```

    **RISKY (1:many):**
    ```
    customers → orders (1:many) ⚠️
    ```

    **DEADLY (fan-trap):**
    ```
    orders → order_items (1:many) + orders → shipments (1:many) 🚨
    ```

    **IF YOUR QUERY INCLUDES THE DEADLY PATTERN:**
    → STOP! Rewrite using UNION approach or separate aggregation CTEs

    ## 🔧 EMERGENCY FAN-TRAP FIX

    If you suspect fan-trap in existing query:
    1. **Split into UNION approach** (recommended)
    2. **Use separate aggregations**
    3. **Add DISTINCT in SUM()** as temporary fix
    4. **Validate results** against source tables
    5. **Always aggregate fact tables separately** before joining

    **Remember:** Fan-traps cause SILENT DATA CORRUPTION! Your query will execute successfully but return WRONG RESULTS. The bigger the multiplication factor, the more wrong your data becomes.

    ## ⚡ AUTOMATED CHECK

    If your query involves more than 2 tables and includes SUM/COUNT/AVG, you MUST analyze for fan-traps before execution. No exceptions!

    ## 🎯 SUCCESS CRITERIA

    Only proceed with `execute_sql_query()` after ALL checks pass:
    - [ ] Schema analyzed ✓
    - [ ] Relationships analyzed ✓  
    - [ ] Fan-trap patterns checked ✓
    - [ ] Syntax validated ✓
    - [ ] Safe aggregation pattern used ✓
    - [ ] Results make business sense ✓

    ## Security Restrictions

    **ALLOWED:**
    - SELECT statements
    - Common Table Expressions (WITH)
    - EXPLAIN statements
    - Database metadata queries

    **PROHIBITED:**
    - INSERT, UPDATE, DELETE statements
    - DDL operations (CREATE, DROP, ALTER)
    - Transaction control (COMMIT, ROLLBACK)
    - System functions that modify state
    - Dynamic SQL execution

    ## Performance Guidelines

    **Query Optimization:**
    - Use appropriate indexes via schema analysis
    - Limit result sets with WHERE clauses
    - Use EXPLAIN to understand query plans
    - Monitor execution time warnings

    **Resource Management:**
    - Default limit: 1000 rows
    - Maximum limit: 5000 rows  
    - Automatic timeout protection
    - Memory usage monitoring

    ## Error Handling

    **Common Error Types:**
    - **Syntax errors**: Use `validate_sql_syntax()` first
    - **Permission errors**: Check allowed query types
    - **Timeout errors**: Simplify complex queries
    - **Memory errors**: Reduce result set size

    **Best Practices:**
    - Always validate syntax before execution
    - Start with small result sets
    - Use LIMIT clauses appropriately
    - Monitor execution time and warnings

    ## Examples

    ### Multi-Fact Query (Recommended)
    ```sql
    WITH unified_facts AS (
        SELECT customer_id, product_id, sales_amount, 0 as returns, 'SALES' as type
        FROM sales
        UNION ALL
        SELECT customer_id, product_id, 0, return_amount, 'RETURNS' as type  
        FROM returns r JOIN sales s ON r.sales_id = s.id
    )
    SELECT customer_id, SUM(sales_amount) as net_sales, SUM(returns) as total_returns
    FROM unified_facts GROUP BY customer_id;
    ```

    ### Safe Aggregation Query
    ```sql
    WITH customer_sales AS (
        SELECT customer_id, SUM(amount) as total_sales
        FROM sales GROUP BY customer_id
    )
    SELECT c.name, cs.total_sales
    FROM customers c
    LEFT JOIN customer_sales cs ON c.id = cs.customer_id
    ORDER BY cs.total_sales DESC;
    ```

    Args:
        sql_query: SQL query to execute (must pass validation first)
        limit: Maximum number of rows to return (default: 1000, max: 5000)
        
    Returns:
        Dictionary with query results and execution metadata
    """
    try:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established", 
                "connection_error",
                "Use connect_database tool first"
            )
        
        result = db_manager.execute_sql_query(sql_query, limit)
        if result['success']:
            logger.info(f"SQL query executed successfully: {result.get('row_count', 0)} rows returned")
        else:
            logger.warning(f"SQL query failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return create_error_response(
            f"Failed to execute SQL query: {str(e)}",
            "execution_error"
        )