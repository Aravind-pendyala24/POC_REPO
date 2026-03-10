/* ============================================================
1. CHECK TOTAL SIZE OF ALL DATABASES
This shows logical storage used by each database.
============================================================ */

SELECT 
    datname AS database_name,
    pg_size_pretty(pg_database_size(datname)) AS database_size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;



/* ============================================================
2. CHECK CURRENT DATABASE SIZE
Useful if you are connected to a specific database.
============================================================ */

SELECT 
    current_database() AS database_name,
    pg_size_pretty(pg_database_size(current_database())) AS db_size;



/* ============================================================
3. BREAKDOWN OF TABLES VS INDEXES STORAGE
Helps determine if indexes are consuming excessive space.
============================================================ */

SELECT
    pg_size_pretty(SUM(pg_relation_size(relid))) AS tables_size,
    pg_size_pretty(SUM(pg_indexes_size(relid))) AS indexes_size,
    pg_size_pretty(SUM(pg_total_relation_size(relid))) AS total_relation_size
FROM pg_catalog.pg_statio_user_tables;



/* ============================================================
4. TOP 20 LARGEST TABLES INCLUDING INDEXES
Useful for identifying large objects consuming storage.
============================================================ */

SELECT
    schemaname,
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_indexes_size(relid)) AS index_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 20;



/* ============================================================
5. WAL (WRITE-AHEAD LOG) STORAGE USAGE
Large WAL size may indicate replication lag or heavy writes.
============================================================ */

SELECT 
    pg_size_pretty(SUM(size)) AS wal_directory_size
FROM pg_ls_waldir();



/* ============================================================
6. TEMPORARY FILE USAGE STATISTICS
Shows cumulative temp file usage since last statistics reset.
Large values indicate queries spilling to disk.
============================================================ */

SELECT
    datname,
    temp_files,
    pg_size_pretty(temp_bytes) AS temp_space_used
FROM pg_stat_database
ORDER BY temp_bytes DESC;



/* ============================================================
7. CHECK DEAD TUPLES (TABLE BLOAT INDICATOR)
High dead tuples mean vacuum may be required.
============================================================ */

SELECT
    relname AS table_name,
    n_live_tup AS live_rows,
    n_dead_tup AS dead_rows
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC
LIMIT 20;



/* ============================================================
8. CHECK UNUSED INDEXES
Indexes with idx_scan = 0 may be unused and removable.
============================================================ */

SELECT
    schemaname,
    relname AS table_name,
    indexrelname AS index_name,
    idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC
LIMIT 20;



/* ============================================================
9. CHECK ACTIVE QUERIES THAT MAY CREATE TEMP FILES
Long-running queries often generate temporary files.
============================================================ */

SELECT
    pid,
    usename,
    state,
    query_start,
    query
FROM pg_stat_activity
WHERE state <> 'idle'
ORDER BY query_start;
