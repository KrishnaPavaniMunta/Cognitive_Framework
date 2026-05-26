import sqlite3
from pathlib import Path

def init_spatial_memory_db(db_path: str = "hospital_twin.db"):
    db_file = Path(db_path)
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS spatial_memory (
            timestamp TEXT NOT NULL,
            class_name TEXT NOT NULL,
            tracker_id INTEGER NOT NULL,
            X REAL NOT NULL,
            Y REAL NOT NULL,
            Z REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_spatial_memory_db()
    print("Database and table created.")
