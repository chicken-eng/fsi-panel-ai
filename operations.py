import re
import concurrent.futures
import pandas as pd
import streamlit as st
from sqlalchemy import text
from database import get_db

def get_all_project_sqls(project_number):
    """
    Fetches distinct semantic fingerprint groups for this project, ordered by first
    appearance. Each entry is {fingerprint, first_sql} where first_sql is the earliest
    stored SQL for that fingerprint (used for count queries and display).
    """
    db = get_db()
    pn_clean = str(project_number).lower()

    query = text("""
        WITH first_occurrence AS (
            SELECT DISTINCT ON (filter_fingerprint)
                filter_fingerprint,
                filters        AS first_sql,
                datetimestamp  AS first_seen
            FROM export_tracker
            WHERE TRIM(LOWER(project_number)) = :pn
              AND filters IS NOT NULL
              AND filter_fingerprint IS NOT NULL
            ORDER BY filter_fingerprint, datetimestamp ASC
        )
        SELECT filter_fingerprint, first_sql
        FROM first_occurrence
        ORDER BY first_seen ASC
    """)

    try:
        with db.engine.connect() as conn:
            rows = conn.execute(query, {"pn": pn_clean}).fetchall()
        return [{"fingerprint": row[0], "first_sql": row[1]} for row in rows]
    except Exception as e:
        st.warning(f"Could not retrieve base SQLs: {e}")
        return []

def clean_sql_for_counts(sql):
    """Removes trailing semicolons and chops off any LIMIT clause at the end."""
    sql = sql.strip().rstrip(";")
    # Regex: Matches 'LIMIT ' followed by numbers at the very end of the string
    sql = re.sub(r'(?i)\s+LIMIT\s+\d+\s*$', '', sql).strip()
    return sql

_DOB_AGE_RE = re.compile(
    r'EXTRACT\s*\(\s*YEAR\s+FROM\s+AGE\s*\(\s*NOW\s*\(\s*\)\s*,\s*\w+\.date_of_birth\s*\)\s*\)\s*'
    r'(?:BETWEEN\s+\d+\s+AND\s+\d+|(?:<=|>=|<|>|!=)\s*\d+)',
    re.IGNORECASE
)

_UPDATE_DATE_RE = re.compile(
    r'(?:\w+\.)?update_date\s*(?:BETWEEN\s+\S+\s+AND\s+\S+|(?:<=|>=|<|>|!=)\s*\S+)',
    re.IGNORECASE
)

def extract_dob_age_condition(sql: str) -> str | None:
    """
    Extracts the EXTRACT(YEAR FROM AGE(NOW(), date_of_birth)) condition
    from a SQL string. Returns the full condition string or None if absent.
    """
    match = _DOB_AGE_RE.search(sql)
    return match.group(0) if match else None

def extract_update_date_condition(sql: str) -> str | None:
    """
    Extracts the update_date comparison condition from a SQL string.
    Returns the condition string or None if absent.
    """
    match = _UPDATE_DATE_RE.search(sql)
    return match.group(0) if match else None

def calculate_project_metrics_cached(project_number):
    """
    Concurrently computes Whole, Launched, Available, Drifted, and Batch metrics
    for each semantic audience fingerprint group registered for this project.

    Key semantics:
      - Whole    : total respondents currently matching the filter, no exclusions.
      - Launched : respondents exported specifically under this TAD's fingerprint.
      - Available: respondents in this TAD's pool NOT yet exported for this project
                   under ANY fingerprint — correctly reflects cross-TAD overlap.
      - Drifted  : exported respondents who no longer satisfy date-based conditions.
    """
    db = get_db()
    tad_groups = get_all_project_sqls(project_number)

    if not tad_groups:
        return []

    pn_clean = str(project_number).lower().strip()

    # Worker definitions — closed over `db`, defined once outside the loop
    def fetch_scalar_worker(sql_str, params):
        with db.engine.connect() as conn:
            return conn.execute(text(sql_str), params).scalar() or 0

    def fetch_dataframe_worker(sql_str, params):
        with db.engine.connect() as conn:
            res = conn.execute(text(sql_str), params)
            return pd.DataFrame(res.fetchall(), columns=res.keys())

    compiled_tasks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for group in tad_groups:
            fingerprint = group["fingerprint"]
            first_sql   = group["first_sql"]
            clean_sql   = clean_sql_for_counts(first_sql)

            # ── Whole: full pool matching these filters, no exclusions ─────────
            query_whole = (
                f"SELECT COUNT(DISTINCT TRIM(LOWER(email))) "
                f"FROM ({clean_sql}) AS sub_whole"
            )

            # ── Launched: exported specifically under this fingerprint ─────────
            query_launched = """
                SELECT COUNT(DISTINCT TRIM(LOWER(email)))
                FROM export_tracker
                WHERE TRIM(LOWER(project_number)) = :pn
                  AND filter_fingerprint = :fingerprint
                  AND email IS NOT NULL
            """

            # ── Available: in pool AND not exported for project under any TAD ──
            query_available = (
                f"SELECT COUNT(DISTINCT TRIM(LOWER(email))) "
                f"FROM ({clean_sql}) AS sub_avail "
                f"WHERE TRIM(LOWER(email)) NOT IN ("
                f"    SELECT TRIM(LOWER(email)) FROM export_tracker "
                f"    WHERE TRIM(LOWER(project_number)) = :pn AND email IS NOT NULL"
                f")"
            )

            # ── Batches: per-day export counts for this fingerprint ────────────
            query_batches = """
                SELECT
                    datetimestamp                          AS export_timestamp,
                    COUNT(DISTINCT TRIM(LOWER(email)))     AS exported
                FROM export_tracker
                WHERE TRIM(LOWER(project_number)) = :pn
                  AND filter_fingerprint = :fingerprint
                  AND email IS NOT NULL
                GROUP BY datetimestamp
                ORDER BY datetimestamp
            """

            # ── Drift detection ───────────────────────────────────────────────
            age_condition    = extract_dob_age_condition(clean_sql)
            update_condition = extract_update_date_condition(clean_sql)
            drift_parts = []
            if age_condition:
                drift_parts.append(f"NOT ({age_condition})")
            if update_condition:
                drift_parts.append(f"NOT ({update_condition})")
            has_date_drift = bool(drift_parts)

            query_drifted = None
            if has_date_drift:
                drift_filter = " OR ".join(drift_parts)
                query_drifted = f"""
                    SELECT COUNT(DISTINCT TRIM(LOWER(et.email)))
                    FROM export_tracker et
                    JOIN respondent r
                        ON TRIM(LOWER(r.email)) = TRIM(LOWER(et.email))
                    WHERE TRIM(LOWER(et.project_number)) = :pn
                      AND et.filter_fingerprint = :fingerprint
                      AND et.email IS NOT NULL
                      AND ({drift_filter})
                """

            fp_params = {"pn": pn_clean, "fingerprint": fingerprint}

            future_whole     = executor.submit(fetch_scalar_worker,    query_whole,     {})
            future_launched  = executor.submit(fetch_scalar_worker,    query_launched,  fp_params)
            future_available = executor.submit(fetch_scalar_worker,    query_available, {"pn": pn_clean})
            future_batches   = executor.submit(fetch_dataframe_worker, query_batches,   fp_params)
            future_drifted   = (
                executor.submit(fetch_scalar_worker, query_drifted, fp_params)
                if query_drifted else None
            )

            compiled_tasks.append({
                "display_sql":    clean_sql,
                "has_date_drift": has_date_drift,
                "futures": {
                    "whole":     future_whole,
                    "launched":  future_launched,
                    "available": future_available,
                    "batches":   future_batches,
                    "drifted":   future_drifted,
                },
            })

    processed_audiences = []
    for task in compiled_tasks:
        whole_count     = task["futures"]["whole"].result()
        launched_count  = task["futures"]["launched"].result()
        available_count = task["futures"]["available"].result()
        batch_df        = task["futures"]["batches"].result()
        drifted_count   = task["futures"]["drifted"].result() if task["futures"]["drifted"] else 0

        processed_audiences.append({
            "display_sql":    task["display_sql"],
            "has_date_drift": task["has_date_drift"],
            "whole_count":    whole_count,
            "launched_count": launched_count,
            "available_count": available_count,
            "drifted_count":  drifted_count,
            "batch_df":       batch_df,
            "pool_shrunk":    whole_count < launched_count,
        })

    return processed_audiences


@st.dialog("Project Action Console")
def open_action_popup(row_data):
    st.write(f"### Management Panel")
    
    project_number = row_data.get('project_number', 'Unknown')
    st.markdown(f"**Target Project:** `{project_number}`")
    st.divider()
    
    try:
        # Pull data instantly from cache. If it's a cache miss, the execution spinner shows once.
        audiences = calculate_project_metrics_cached(project_number)
        
        if not audiences:
            st.info("No saved filters/exports found for this project yet. Launch a sample from the FSI AI page to populate these metrics.")
            return

        # Render UI components without executing ANY underlying database operations
        for i, data in enumerate(audiences):
            if len(audiences) > 1:
                st.markdown(f"#### Target Audience Definition #{i + 1}")
                
            if data["has_date_drift"]:
                m1, m2, m3, m4 = st.columns(4)
                m4.metric(
                    "Drifted Out",
                    f"{data['drifted_count']:,}",
                    delta=f"-{data['drifted_count']:,}" if data['drifted_count'] > 0 else None,
                    delta_color="inverse",
                    help="Exported respondents who no longer satisfy the age or date-based criteria in the original query."
                )
            else:
                m1, m2, m3 = st.columns(3)
                
            m1.metric("Whole Sample", f"{data['whole_count']:,}")
            m2.metric("Launched Sample", f"{data['launched_count']:,}")
            m3.metric("Available Sample", f"{data['available_count']:,}")

            if data["pool_shrunk"]:
                st.warning(
                    f"⚠️ Pool has shrunk below launched count — "
                    f"{data['launched_count']:,} exported but only {data['whole_count']:,} "
                    "currently qualify. Investigate before launching further."
                )
            
            batch_df = data["batch_df"]
            if not batch_df.empty:
                batch_df = batch_df.rename(columns={
                    "export_timestamp": "Export Date",
                    "exported":    "Emails Exported"
                })
                batch_df["Export Date"] = (
                    pd.to_datetime(batch_df["Export Date"])
                    .dt.strftime("%d %b %Y")
                )

                st.markdown("**Export Batches**")
                for idx, row in batch_df.iterrows():
                    st.markdown(f"📅 **Export Date:** `{row['Export Date']}`   |   ✉️ **Emails Exported:** `{row['Emails Exported']:,}`")
                    
            with st.expander("View Base Target Parameters (SQL)", expanded=False):
                st.code(data["display_sql"], language="sql")

            if i < len(audiences) - 1:
                st.divider()

    except Exception as e:
        st.error("Failed to compile or display sample calculations.")
        with st.expander("View Error Details"):
            st.exception(e)


def show_operations_page():
    st.title("Operations Activity Logs")
    st.subheader("Open projects")
    
    db = get_db()
    query = "SELECT project_number, project_name, project_type, topic, sharepoint_link, created_date FROM projects WHERE project_state = 'Open' order by project_type;"

    try:
        df = db.get_df(query)
        
        if not df.empty:
            st.caption("💡 Highlight any row below to trigger the operations action popup.")
            
            selection = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row" 
            )
            
            if selection and selection.get("selection", {}).get("rows"):
                selected_row_idx = selection["selection"]["rows"][0]
                row_data = df.iloc[selected_row_idx].to_dict()
                
                open_action_popup(row_data)
                
        else:
            st.info("There are currently no projects marked as 'Open'.")
            
    except Exception as e:
        st.error(f"Failed to load operational project metrics: {e}")

    st.subheader("New respondents per project")

    query2 = """WITH first_seen AS (
                   SELECT 
                       email,
                       MIN(created_date) AS first_created_date
                   FROM respondent_type_specification
                   GROUP BY email
                   HAVING MIN(created_date) >= DATE '2026-01-01'
                )
            
                SELECT
                    COALESCE(p.project_number::text, 'TOTAL') AS project_number,
                    COALESCE(p.project_name, 'TOTAL') AS project_name,
                    COUNT(DISTINCT fs.email) AS new_respondents
                FROM first_seen fs
                JOIN project_respondent pr
                   ON pr.email = fs.email
                   AND pr.last_activity_date = fs.first_created_date
                JOIN projects p
                   ON p.project_number = pr.project_number
                GROUP BY ROLLUP (
                    p.project_number,
                    p.project_name
                 )
                 ORDER BY
                     CASE WHEN p.project_number IS NULL THEN 1 ELSE 0 END,
                     new_respondents DESC;
    """

    try:
        df2 = db.get_df(query2)

        if not df2.empty:
            st.dataframe(
                df2,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No data available for this metric.")

    except Exception as e:
        st.error(f"Failed to load secondary metrics: {e}")
