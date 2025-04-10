# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- File Paths ---
# Use absolute paths for more robustness if needed, but relative should work fine.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets directory where config.py is
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
POLICY_INDEX_DIR = os.path.join(BASE_DIR, "policy_indices")
MANIFEST_FILE = os.path.join(POLICY_INDEX_DIR, "manifest.json")
DB_FILE = os.path.join(BASE_DIR, "expense_analysis.db")
RECEIPT_STORAGE_DIR = os.path.join(BASE_DIR, "receipt_storage") # <<< NEW PATH

# --- Model Names ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-4o" # Keep the model we switched to

# --- Application Settings ---
RESULTS_PER_PAGE = 15

# --- Ensure Directories Exist ---
# Moved here to run once when config is imported
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(POLICY_INDEX_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECEIPT_STORAGE_DIR, exist_ok=True) # <<< ENSURE IT'S CREATED
# --- Optional: Check for API Key ---
# Can add a check here, but maybe better handled when LLM is initialized
# if not OPENAI_API_KEY:
#    print("WARNING: OPENAI_API_KEY not found in environment variables.")