import sys
import os
import json
import psycopg2


def print_summary(environment, sql_file, status, error=None):
    summary = {
        "environment": environment,
        "file": os.path.basename(sql_file),
        "status": status
    }

    if error:
        summary["error"] = error

    print(json.dumps(summary, indent=2))


def main():
    if len(sys.argv) != 7:
        print("Invalid arguments supplied.")
        # Still exit 0 so Terraform doesn't fail
        sys.exit(0)

    environment = sys.argv[1]
    host = sys.argv[2]
    dbname = sys.argv[3]
    user = sys.argv[4]
    password = sys.argv[5]
    sql_file = sys.argv[6]

    if not os.path.exists(sql_file):
        print_summary(environment, sql_file, "failed", "SQL file not found")
        sys.exit(0)   # 🔥 DO NOT FAIL TERRAFORM

    try:
        conn = psycopg2.connect(
            host=host,
            database=dbname,
            user=user,
            password=password
        )
        conn.autocommit = False
        cursor = conn.cursor()

        with open(sql_file, "r") as file:
            sql = file.read()
            cursor.execute(sql)

        conn.commit()
        cursor.close()
        conn.close()

        print_summary(environment, sql_file, "success")

    except Exception as e:
        print_summary(environment, sql_file, "failed", str(e))

    # 🔥 Always exit 0
    sys.exit(0)


if __name__ == "__main__":
    main()
