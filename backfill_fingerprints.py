#!/usr/bin/env python3
"""
One-time migration script — populates filter_fingerprint for all existing
export_tracker rows where it is currently NULL.

Run AFTER the ALTER TABLE migration and BEFORE deploying the updated operations.py.

Usage:
    python backfill_fingerprints.py
"""

import re
import hashlib
from database import get_db
from sqlalchemy import text


# ── Duplicated from fsi_ai.py to keep this script self-contained ─────────────

def _extract_base_sql(sql: str) -> str:
    """Strips the export-exclusion outer wrapper, returning the inner base SQL."""
    match = re.search(
        r"SELECT \* FROM \(\s*(.*?)\s*\)\s*AS final_output\s*WHERE final_output\.email NOT IN",
        sql,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else sql.strip()


def compute_filter_fingerprint(sql: str) -> str:
    """
    Produces a stable 16-char fingerprint from the semantic WHERE conditions
    of a filter SQL string.  Two queries with identical demographic requirements
    but differing SELECT columns, ORDER BY, JOIN style, or whitespace will
    produce the same fingerprint and are treated as one Target Audience Definition.
    """
    base_sql = _extract_base_sql(sql)

    # Drop ORDER BY and everything trailing it — irrelevant to semantics
    base_no_order = re.sub(
        r"\bORDER\s+BY\b.*$", "", base_sql, flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # Extract just the WHERE clause
    where_match = re.search(r"\bWHERE\b(.+)$", base_no_order, re.DOTALL | re.IGNORECASE)
    where_clause = where_match.group(1).strip() if where_match else base_no_order

    # Remove the universal blacklist exclusion — present in every query, adds no signal
    where_clause = re.sub(
        r"\s*AND\s+r\.email\s+NOT\s+IN\s*\(\s*SELECT\s+email\s+FROM\s+unsubscribe_blacklist\s*\)",
        "",
        where_clause,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # Normalize: collapse whitespace → single space, lowercase, strip table alias prefixes
    normalized = re.sub(r"\s+", " ", where_clause).lower().strip()
    normalized = re.sub(r"\b(?:r|a|rts|rt)\b\.", "", normalized)

    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────

def main():
    db = get_db()

    with db.engine.connect() as conn:
        # Fetch all distinct filters that haven't been fingerprinted yet
        rows = conn.execute(
            text("""
                SELECT DISTINCT filters
                FROM export_tracker
                WHERE filter_fingerprint IS NULL
                  AND filters IS NOT NULL
            """)
        ).fetchall()

        if not rows:
            print("Nothing to backfill — all rows already have a fingerprint.")
            return

        print(f"Found {len(rows)} distinct SQL string(s) to fingerprint...")

        updated_total = 0
        for (filters_sql,) in rows:
            try:
                fp = compute_filter_fingerprint(filters_sql)
                result = conn.execute(
                    text("""
                        UPDATE export_tracker
                        SET filter_fingerprint = :fp
                        WHERE filters = :sql
                          AND filter_fingerprint IS NULL
                    """),
                    {"fp": fp, "sql": filters_sql},
                )
                updated_total += result.rowcount
                print(f"  fingerprint={fp}  rows_updated={result.rowcount}")
            except Exception as e:
                print(f"  ERROR computing fingerprint for SQL snippet: {e}")

        conn.commit()

    print(f"\nBackfill complete. {updated_total} row(s) updated.")


if __name__ == "__main__":
    main()
