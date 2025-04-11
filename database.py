# database.py
import sqlite3
import streamlit as st
import os
import traceback
# Import DB path from config
from config import DB_FILE

def init_db():
    """Initializes the SQLite database and updates the table schema if needed."""
    conn = None
    db_dir = os.path.dirname(DB_FILE)
    if db_dir and not os.path.exists(db_dir):
         try: os.makedirs(db_dir)
         except OSError as e: st.error(f"FATAL: Could not create DB directory {db_dir}: {e}"); st.stop()

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
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
        conn.commit()

        cursor.execute("PRAGMA table_info(receipt_analysis);")
        existing_columns = [column[1] for column in cursor.fetchall()]

        if 'merchant_name' not in existing_columns:
            try: cursor.execute("ALTER TABLE receipt_analysis ADD COLUMN merchant_name TEXT"); conn.commit()
            except sqlite3.OperationalError as e: print(f"Info: merchant_name column check: {e}")
        if 'total_amount' not in existing_columns:
            try: cursor.execute("ALTER TABLE receipt_analysis ADD COLUMN total_amount TEXT"); conn.commit()
            except sqlite3.OperationalError as e: print(f"Info: total_amount column check: {e}")
        if 'stored_file_path' not in existing_columns:
            try: cursor.execute("ALTER TABLE receipt_analysis ADD COLUMN stored_file_path TEXT"); conn.commit()
            except sqlite3.OperationalError as e: print(f"Info: stored_file_path column check: {e}")

    except sqlite3.Error as e: st.error(f"FATAL: DB Init/Schema Error: {e}"); st.error(traceback.format_exc()); st.stop()
    finally:
        if conn: conn.close()

def check_duplicate(receipt_hash: str):
    """Checks DB for duplicate hash. Returns is_duplicate, status, timestamp."""
    conn = None;
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT status, analysis_timestamp FROM receipt_analysis WHERE receipt_hash = ? ORDER BY analysis_timestamp DESC LIMIT 1", (receipt_hash,))
        result = cursor.fetchone()
        return (True, result[0], result[1]) if result else (False, None, None)
    except sqlite3.Error as e: st.error(f"DB Duplicate Check Error: {e}"); return False, None, None
    finally:
        if conn: conn.close()

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


# <<< MODIFIED get_historical_results >>>
def get_historical_results(limit: int, offset: int, status_filter: list = None):
    """Fetches historical results from DB with pagination and optional status filter."""
    conn = None
    results = []
    # Base query
    query = """
        SELECT receipt_filename, status, reasoning, policy_name_used,
               analysis_timestamp, merchant_name, total_amount, stored_file_path
        FROM receipt_analysis
    """
    params = [] # Parameters for SQL query

    # --- Add WHERE clause if filter is provided ---
    if status_filter:
        # Ensure status_filter is a list and not empty
        if isinstance(status_filter, list) and status_filter:
            placeholders = ', '.join('?' for _ in status_filter)
            query += f" WHERE status IN ({placeholders})"
            params.extend(status_filter)
    # --- End WHERE clause ---

    # Add Ordering, Limit, Offset
    query += " ORDER BY analysis_timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, tuple(params)) # Pass parameters as tuple
        results = [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        st.error(f"DB History Fetch Error (Filter: {status_filter}): {e}")
        st.error(traceback.format_exc()) # Add traceback
    finally:
        if conn:
            conn.close()
    return results
# <<< END MODIFIED FUNCTION >>>


def get_result_by_filename(filename: str):
    """Fetches the latest analysis record for a specific filename."""
    conn = None
    result = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM receipt_analysis WHERE receipt_filename = ? ORDER BY analysis_timestamp DESC LIMIT 1", (filename,))
        row = cursor.fetchone()
        if row: result = dict(row)
    except sqlite3.Error as e: st.error(f"DB Error fetching result for {filename}: {e}"); st.error(traceback.format_exc())
    finally:
        if conn: conn.close()
    return result