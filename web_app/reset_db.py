from app import db, app
import os

# Delete old database if it exists
if os.path.exists("users.db"):
    os.remove("users.db")
    print("Old database deleted.")

# Create a new database inside the application context
with app.app_context():
    db.create_all()
    print("Database recreated successfully!")
