import sys
import os
import json
import psycopg2
from datetime import datetime

DEFAULT_PORT = 5432


def log(message):
    print(f"[{datetime.utcnow().isoformat()}] {message}")


def main():
    if len(sys.argv) < 7:
        log("Invalid arguments supplied.")
        sys.exit(0)

    environment = sys.argv[1]
    host = sys.argv[2]
    dbname = sys.argv[3]
    user = sys.argv[4]
    password = sys.argv[5]

    # Remaining args are SQL files
    sql_files = sys.argv[6:]

    results = []

    conn = None
    cursor = None

    try:
        log(f"Connecting to DB {host}:{DEFAULT_PORT}")
        conn = psycopg2.connect(
            host=host,
            port=DEFAULT_PORT,
            database=dbname,
            user=user,
            password=password
        )
        conn.autocommit = False
        cursor = conn.cursor()

        log("Database connection successful.")

        for sql_file in sql_files:

            if not os.path.exists(sql_file):
                log(f"File not found: {sql_file}")
                results.append({
                    "file": sql_file,
                    "status": "failed",
                    "error": "File not found"
                })
                continue

            try:
                log(f"Executing: {sql_file}")

                with open(sql_file, "r") as file:
                    sql = file.read()
                    cursor.execute(sql)

                log(f"{sql_file} executed successfully.")

                results.append({
                    "file": sql_file,
                    "status": "success"
                })

            except Exception as e:
                log(f"Error executing {sql_file}: {str(e)}")
                conn.rollback()

                results.append({
                    "file": sql_file,
                    "status": "failed",
                    "error": str(e)
                })

        conn.commit()
        log("Transaction committed.")

    except Exception as e:
        log(f"Connection-level failure: {str(e)}")

    finally:
        if cursor:
            cursor.close()
            log("Cursor closed.")
        if conn:
            conn.close()
            log("Connection closed.")

    summary = {
        "environment": environment,
        "results": results
    }

    print("\n--- Migration Summary ---")
    print(json.dumps(summary, indent=2))
    print("-------------------------\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
