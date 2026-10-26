import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"

# Mounted read-only in docker-compose at /app/data_docs (see docs/data_docs in the repo)
DOCUMENTS_DIR = Path(os.environ.get("DOCUMENTS_DIR", "/app/data_docs"))

# Mounted read-write volume for the contact form sqlite db
CONTACT_DB_PATH = Path(os.environ.get("CONTACT_DB_PATH", "/app/storage/contact.db"))

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

# RAG / chat settings
VECTOR_STORE_DIR = os.environ.get("VECTOR_STORE_DIR", "/app/vector_store")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
