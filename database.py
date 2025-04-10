# database.py
import sqlite3
import streamlit as st
import os
import traceback # Add traceback
from config import DB_FILE


def init_db():
    """Initializes the SQLite database and updates the table schema if needed."""
    conn = None
    db_dir = os.path.dirname(DB_FILE)
    if db_dir and not os.path.exists(db_dir):
         try:
             os.makedirs(db_dir)
             print(f"Created database directory: {db_dir}")
         except OSError as e:
             st.error(f"FATAL: Could not create database directory {db_dir}: {e}")
             st.stop()

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Create table if not exists (original schema + new fields for robustness)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS receipt_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_filename TEXT NOT NULL,
                receipt_hash TEXT NOT NULL UNIQUE,
                policy_name_used TEXT NOT NULL,
                analysis_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL CHECK(status IN ('Approved', 'Rejected', 'On Hold', 'Error', 'Duplicate')),
                reasoning TEXT,
                merchant_name TEXT,
                total_amount TEXT,
                stored_file_path TEXT
            )
        """)
        conn.commit() # Commit table creation first

        # Add new columns if they don't exist (Safer for existing DBs)
        # Check existing columns
        cursor.execute("PRAGMA table_info(receipt_analysis);")
        existing_columns = [column[1] for column in cursor.fetchall()]

        if 'merchant_name' not in existing_columns:
            try:
                cursor.execute("ALTER TABLE receipt_analysis ADD COLUMN merchant_name TEXT")
                print("Added merchant_name column")
                conn.commit()
            except sqlite3.OperationalError as e:
                print(f"Could not add merchant_name column (might already exist or other issue): {e}") # More informative

        if 'total_amount' not in existing_columns:
            try:
                cursor.execute("ALTER TABLE receipt_analysis ADD COLUMN total_amount TEXT")
                print("Added total_amount column")
                conn.commit()
            except sqlite3.OperationalError as e:
                 print(f"Could not add total_amount column (might already exist or other issue): {e}")

        if 'stored_file_path' not in existing_columns:
            try:
                cursor.execute("ALTER TABLE receipt_analysis ADD COLUMN stored_file_path TEXT")
                print("Added stored_file_path column")
                conn.commit()
            except sqlite3.OperationalError as e:
                 print(f"Could not add stored_file_path column (might already exist or other issue): {e}")

    except sqlite3.Error as e:
        st.error(f"FATAL: Database Schema Update/Init Error: {e}")
        st.error(traceback.format_exc()) # Add traceback for DB errors
        st.stop()
    finally:
        if conn:
            conn.close()

def check_duplicate(receipt_hash: str):
    """Checks DB for duplicate hash. Returns is_duplicate, status, timestamp."""
    conn = None;
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT status, analysis_timestamp FROM receipt_analysis WHERE receipt_hash = ? ORDER BY analysis_timestamp DESC LIMIT 1", (receipt_hash,))
        result = cursor.fetchone()
        return (True, result[0], result[1]) if result else (False, None, None)
    except sqlite3.Error as e:
        st.error(f"DB Duplicate Check Error: {e}")
        return False, None, None
    finally:
        if conn:
            conn.close()

def save_analysis_result(filename: str, file_hash: str, policy_name: str,
                         status: str, reasoning: str,
                         merchant: str = None, amount: str = None,
                         stored_path: str = None):
    """Saves the analysis result, including stored file path, to the database."""
    conn = None;
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO receipt_analysis
                (receipt_filename, receipt_hash, policy_name_used, status, reasoning,
                 merchant_name, total_amount, stored_file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (filename, file_hash, policy_name, status, reasoning,
              merchant, amount, stored_path))
        conn.commit()
    except sqlite3.IntegrityError: st.warning(f"DB Integrity: Duplicate hash {file_hash[:8]}... for {filename}.")
    except sqlite3.Error as e: st.error(f"DB Save Error for {filename}: {e}")
    finally:
        if conn: conn.close()


def get_historical_results(limit: int, offset: int):
    """Fetches historical analysis results including stored file path."""
    conn = None
    results = []
    required_columns = ["receipt_filename", "status", "reasoning", "policy_name_used",
                        "analysis_timestamp", "merchant_name", "total_amount", "stored_file_path"]
    select_cols_str = ", ".join(required_columns) # Ensure all needed columns are selected

    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Select all required columns explicitly
        cursor.execute(f"""
            SELECT {select_cols_str}
            FROM receipt_analysis
            ORDER BY analysis_timestamp DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        results = [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        st.error(f"DB History Fetch Error: {e}")
        st.error(traceback.format_exc()) # Add traceback
    finally:
        if conn:
            conn.close()
    return results

# <<< NEW FUNCTION >>>
def get_result_by_filename(filename: str):
    """Fetches the latest analysis record for a specific filename."""
    conn = None
    result = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Fetch all columns for the latest entry matching the filename
        cursor.execute("""
            SELECT *
            FROM receipt_analysis
            WHERE receipt_filename = ?
            ORDER BY analysis_timestamp DESC
            LIMIT 1
        """, (filename,))
        row = cursor.fetchone()
        if row:
            result = dict(row) # Convert row to dict
    except sqlite3.Error as e:
        st.error(f"DB Error fetching result for {filename}: {e}")
        st.error(traceback.format_exc())
    finally:
        if conn:
            conn.close()
    return result
# <<< END NEW FUNCTION >>>