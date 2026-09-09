import sqlite3

connection = sqlite3.connect("detections.db")
cursor = connection.cursor()

cursor.execute("PRAGMA table_info(detections)")

columns = cursor.fetchall()

print("Columns in detections table:")

for column in columns:
    print(column)

connection.close()