import sqlite3
from datetime import datetime, timezone

from ..core.config import CONTACT_DB_PATH
from ..schemas import ContactInquiry


def _connect() -> sqlite3.Connection:
    CONTACT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CONTACT_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS contact_inquiries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            organization TEXT,
            inquiry_type TEXT,
            message TEXT NOT NULL
        )
        """
    )
    return conn


def save_contact_inquiry(inquiry: ContactInquiry) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO contact_inquiries
                (created_at, name, email, organization, inquiry_type, message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                inquiry.name,
                inquiry.email,
                inquiry.organization,
                inquiry.inquiryType,
                inquiry.message,
            ),
        )


def list_contact_inquiries() -> list[dict]:
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT id, created_at, name, email, organization, inquiry_type, message "
            "FROM contact_inquiries ORDER BY id DESC"
        )
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
