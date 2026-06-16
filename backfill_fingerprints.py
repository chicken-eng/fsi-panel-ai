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


def _split_top_level_and(clause: str) -> list[str]:
    """
    Splits a normalized (lowercase) WHERE clause on AND conjunctions at
    parenthesis depth 0 only. Correctly protects BETWEEN x AND y from
    being split, and ignores AND inside subexpressions.
    """
    # Step 1 — stash BETWEEN x AND y so its internal AND is invisible to the splitter
    between_slots = {}
    slot_counter = [0]

    def stash_between(m):
        token = f'\x00BTWN{slot_counter[0]}\x00'
        between_slots[token] = m.group(0)
        slot_counter[0] += 1
        return token

    # Matches: BETWEEN <number or 'quoted string'> AND <number or 'quoted string'>
    safe_clause = re.sub(
        r"\bbetween\s+(?:'[^']*'|\S+)\s+and\s+(?:'[^']*'|\S+)",
        stash_between,
        clause
    )

    # Step 2 — depth-aware split on top-level AND
    parts = []
    depth = 0
    start = 0
    i = 0

    while i < len(safe_clause):
        ch = safe_clause[i]
        if ch == '(':
            depth += 1
            i += 1
        elif ch == ')':
            depth -= 1
            i += 1
        elif depth == 0 and safe_clause[i:i+3] == 'and':
            before_ok = (i == 0 or (not safe_clause[i-1].isalnum() and safe_clause[i-1] != '_'))
            after_ok  = (i+3 >= len(safe_clause) or (not safe_clause[i+3].isalnum() and safe_clause[i+3] != '_'))
            if before_ok and after_ok:
                part = safe_clause[start:i].strip()
                if part:
                    parts.append(part)
                start = i + 3
                i += 3
                continue
            i += 1
        else:
            i += 1

    last = safe_clause[start:].strip()
    if last:
        parts.append(last)

    # Step 3 — restore BETWEEN expressions
    restored = []
    for part in parts:
        for token, original in between_slots.items():
            part = part.replace(token, original)
        restored.append(part.strip())

    return [p for p in restored if p]


def _normalize_in_list(condition: str) -> str:
    """
    Sorts values inside IN (...) value lists so that:
    ethnicity IN ('White Irish', 'White British', 'White European')
    ethnicity IN ('White European', 'White British', 'White Irish')
    both produce the same canonical string.
    Only operates on simple value lists — not subqueries (those contain ')').
    """
    def sort_values(m):
        values = [v.strip() for v in m.group(1).split(',')]
        values.sort()
        return f"in ({', '.join(values)})"
    return re.sub(r'\bin\s*\(([^)]+)\)', sort_values, condition)


def compute_filter_fingerprint(sql: str) -> str:
    """
    Produces a stable 16-char fingerprint from the semantic WHERE conditions
    of a filter SQL string, invariant across:

      - SELECT column order, JOIN style, ORDER BY, whitespace
      - Table alias prefixes (r., a., rts., rt., ub.)
      - PostgreSQL type casts (::text, ::varchar, etc.)
      - ILIKE '%value%' vs = 'value'
      - String literal casing
      - Blacklist subquery alias variations
      - WHERE condition ordering  ← new
      - IN list value ordering    ← new
      - BETWEEN x AND y integrity ← new
    """
    # fsi_ai.py uses:        base_sql = extract_base_sql_for_storage(sql)
    # backfill script uses:  base_sql = _extract_base_sql(sql)
    base_sql = _extract_base_sql(sql)

    # Drop ORDER BY and everything trailing it
    base_no_order = re.sub(
        r'\bORDER\s+BY\b.*$', '', base_sql, flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # Extract the WHERE clause body
    where_match = re.search(r'\bWHERE\b(.+)$', base_no_order, re.DOTALL | re.IGNORECASE)
    where_clause = where_match.group(1).strip() if where_match else base_no_order

    # Remove blacklist exclusion in all alias forms
    where_clause = re.sub(
        r'\s*AND\s+\w+\.email\s+NOT\s+IN\s*\(\s*SELECT\s+(?:\w+\.)?email\s+FROM\s+unsubscribe_blacklist(?:\s+(?:AS\s+)?\w+)?\s*\)',
        '', where_clause, flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # Strip PostgreSQL type casts
    where_clause = re.sub(r'::\w+', '', where_clause)

    # Normalize ILIKE '%value%' or ILIKE 'value' → = 'value'
    where_clause = re.sub(
        r"\bILIKE\s+'%?([^'%]+)%?'",
        r"= '\1'",
        where_clause, flags=re.IGNORECASE
    )

    # Lowercase string literals so 'United Kingdom' and 'united kingdom' match
    where_clause = re.sub(r"'[^']*'", lambda m: m.group(0).lower(), where_clause)

    # Collapse all whitespace to single space, then lowercase everything
    where_clause = re.sub(r'\s+', ' ', where_clause).lower().strip()

    # Strip known table alias prefixes
    where_clause = re.sub(r'\b(?:r|a|rts|rt|ub)\b\.', '', where_clause)

    # Split into individual conditions, canonicalize each, sort, rejoin
    conditions = _split_top_level_and(where_clause)
    conditions = [_normalize_in_list(c) for c in conditions]
    conditions.sort()

    canonical = ' and '.join(conditions)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


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
