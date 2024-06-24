import pandas as pd
import sqlite3

# Step 1: Load your CSV file into a DataFrame
df = pd.read_csv('heart.csv')

# Step 2: Create a connection object that represents the database
conn = sqlite3.connect('heart_disease.db')

# Step 3: Use the DataFrame's to_sql method to create a new table in the SQLite database
df.to_sql('heart_disease', conn, if_exists='replace', index=False)

# Step 4: Create a cursor object
cursor = conn.cursor()

# Step 5: Execute a simple query
cursor.execute('SELECT * FROM heart_disease LIMIT 5')

# Fetch and print the results
rows = cursor.fetchall()
for row in rows:
    print(row)

# Step 6: Close the connection when done
conn.close()
