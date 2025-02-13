from app import db

# Delete old database file (if exists)
import os
if os.path.exists("users.db"):
    os.remove("users.db")

# Create new database schema
db.create_all()
print("Database recreated successfully!")
