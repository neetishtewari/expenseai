# utils.py
import hashlib

def calculate_hash(file_content: bytes) -> str:
    """Calculates the SHA-256 hash of the file content."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_content)
    return sha256_hash.hexdigest()

# We could move extract_text_from_file here later, but it has many dependencies.
# Keeping it in analysis.py for now is simpler.