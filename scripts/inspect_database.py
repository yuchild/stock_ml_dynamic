from sqlalchemy import inspect, text
from stock_ml_dynamic.db import engine, qualified_table

def main():
    with engine().connect() as conn:
        inspector = inspect(conn)
        print("Database connection: OK")
        for schema in ("raw", "analytics", "mart"):
            if schema in inspector.get_schema_names():
                print(f"\n{schema} tables:", ", ".join(inspector.get_table_names(schema=schema)))
        schema, table = qualified_table()
        print(f"\n{schema}.{table} columns:")
        for col in inspector.get_columns(table, schema=schema):
            print(f"  {col['name']}: {col['type']}")
        print("\nNo data modified.")

if __name__ == "__main__":
    main()
