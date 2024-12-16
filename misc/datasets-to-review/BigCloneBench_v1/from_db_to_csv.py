import psycopg2
import csv
import os

# Database connection parameters
db_params = {
    'dbname': 'bcb',
    'user': 'postgres',  # Default superuser
    'password': '123',  # Replace with your actual password
    'host': 'localhost',
    'port': '5432'
}

# Connect to the PostgreSQL database
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

# Get the list of tables
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public';
""")
tables = cursor.fetchall()

# Directory to save CSV files
output_dir = 'csv_exports'
os.makedirs(output_dir, exist_ok=True)

# Export each table to a CSV file
for table_name_tuple in tables:
    table_name = table_name_tuple[0]
    csv_file_path = os.path.join(output_dir, f"{table_name}.csv")
    
    # Execute query to get table data
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    # Get column names
    column_names = [desc[0] for desc in cursor.description]
    
    # Write data to CSV with UTF-8 encoding
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)  # Write column headers
        csv_writer.writerows(rows)  # Write rows
    
    print(f"Exported {table_name} to {csv_file_path}")

# Close the cursor and connection
cursor.close()
conn.close()
