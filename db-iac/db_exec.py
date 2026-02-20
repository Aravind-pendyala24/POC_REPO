import os
import psycopg2
from datetime import datetime
import sys

DELIMITER = "|"

def log(msg):
    print(f"[{datetime.utcnow().isoformat()}] {msg}")


def get_sql_files():
    sql_files_env = os.getenv("SQL_FILES", "")
    base_path = os.getenv("SQL_BASE_PATH", "")

    if not sql_files_env:
        log("No SQL_FILES provided.")
        return []

    files = [f.strip() for f in sql_files_env.split(DELIMITER) if f.strip()]

    resolved_files = []

    for f in files:
        try:
            full_path = (
                os.path.normpath(os.path.join(base_path, f))
                if base_path else f
            )
            resolved_files.append(full_path)
        except Exception as e:
            log(f"Error resolving file path {f}: {str(e)}")

    return resolved_files


def main():

    environment = os.getenv("ENVIRONMENT")
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    sql_files = get_sql_files()

    if not sql_files:
        log("No valid SQL files to execute.")
        sys.exit(0)

    try:
        log("Connecting to database...")

        conn = psycopg2.connect(
            host=host,
            port=port,
            database=dbname,
            user=username,
            password=password,
            sslmode="require"
        )

        conn.autocommit = False
        cursor = conn.cursor()

        log("Database connection established successfully.")

        results = []

        for sql_file in sql_files:

            if not os.path.exists(sql_file):
                log(f"File not found: {sql_file}")
                results.append({
                    "file": os.path.basename(sql_file),
                    "status": "failed",
                    "error": "File not found"
                })
                continue

            try:
                log(f"Executing {sql_file}")

                with open(sql_file, "r") as f:
                    cursor.execute(f.read())

                results.append({
                    "file": os.path.basename(sql_file),
                    "status": "success"
                })

            except Exception as e:
                conn.rollback()
                log(f"Error executing {sql_file}: {str(e)}")

                results.append({
                    "file": os.path.basename(sql_file),
                    "status": "failed",
                    "error": str(e)
                })

        conn.commit()
        cursor.close()
        conn.close()

        log("Database connection closed.")

        print("\n--- Migration Summary ---")
        print({
            "environment": environment,
            "results": results
        })
        print("-------------------------\n")

    except Exception as e:
        log(f"Fatal error: {str(e)}")

    # Ensure Terraform never fails
    sys.exit(0)


if __name__ == "__main__":
    main()
