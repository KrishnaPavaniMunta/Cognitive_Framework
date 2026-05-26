import sqlite3
db_path = r"d:\Object Detection Model\yolo_tr\yolo_tr\hospital_detector_longterm\rgbd_development\output\hospital_twin.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
print("--- SCHEMA ---")
cursor.execute("PRAGMA table_info(spatial_memory)")
for col in cursor.fetchall(): print(col)
print("\n--- LATEST SESSION IDS ---")
cursor.execute("SELECT DISTINCT session_id FROM spatial_memory ORDER BY session_id DESC LIMIT 5")
print(cursor.fetchall())
print("\n--- RECENT ROWS ---")
query = "SELECT timestamp, class_name, tracker_id, X, Y, Z, last_seen, session_id FROM spatial_memory WHERE class_name IN (\"chair\", \"laptop\", \"bottle\") ORDER BY timestamp DESC LIMIT 20"
cursor.execute(query)
for row in cursor.fetchall(): print(row)
conn.close()
