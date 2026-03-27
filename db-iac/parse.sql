INSERT INTO my_table (col1, col2)
SELECT *
FROM (
    VALUES
        ('A', 'X'),
        ('B', 'Y')
) AS v(col1, col2)
WHERE NOT EXISTS (
    SELECT 1
    FROM my_table t
    WHERE t.col1 = v.col1
      AND t.col2 = v.col2
);
