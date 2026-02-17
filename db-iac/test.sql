BEGIN;

CREATE TABLE IF NOT EXISTS tf_dummy_test (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50)
);

INSERT INTO tf_dummy_test (name)
SELECT 'terraform_test'
WHERE NOT EXISTS (
    SELECT 1 FROM tf_dummy_test WHERE name='terraform_test'
);

SELECT * FROM tf_dummy_test;

COMMIT;
