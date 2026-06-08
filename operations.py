import re
import concurrent.futures
import pandas as pd
import streamlit as st
from sqlalchemy import text
from database import get_db

def get_all_project_sqls(project_number):
    """Fetches ALL distinct saved AI queries for this project from the export tracker."""
    db = get_db()
    query = """
        SELECT filters 
        FROM export_tracker 
        WHERE project_number = :pn AND filters IS NOT NULL 
        GROUP BY filters 
        ORDER BY MIN(datetimestamp) ASC
    """
    try:
        # Wrap query string in text() for SQLAlchemy 2.0 conformance
        result = db.execute(query, {"pn": str(project_number).lower()})
        return [row[0] for row in result]
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


@st.cache_data(ttl=300, show_spinner="Calculating live sample metrics concurrently...")
def calculate_project_metrics_cached(project_number):
    """
    Performance Core: Extracts, prepares, and runs all target data counters 
    simultaneously across independent connection channels via a ThreadPoolExecutor.
    Caches outputs for 5 minutes to prevent heavy interactive UI re-query penalties.
    """
    db = get_db()
    base_sqls = get_all_project_sqls(project_number)
    
    if not base_sqls:
        return []

    pn_clean = str(project_number).lower().strip()
    compiled_tasks = []
    
    # Open thread pool context manager to manage concurrent database lookups
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        for i, raw_sql in enumerate(base_sqls):
            clean_sql = clean_sql_for_counts(raw_sql)
            
            # Setup isolation representations 
            if i == 0:
                display_sql = clean_sql
                query_whole = f"SELECT COUNT(DISTINCT TRIM(LOWER(email))) FROM ({clean_sql}) AS sub_whole"
            else:
                display_sql = (
                    f"SELECT * FROM (\n"
                    f"    {clean_sql}\n"
                    f") AS final_output\n"
                    f"WHERE final_output.email NOT IN (\n"
                    f"    SELECT email \n"
                    f"    FROM export_tracker\n"
                    f"    WHERE project_number = '{pn_clean}'\n"
                    f"    AND filters != '<< THIS CURRENT DEMOGRAPHIC QUERY >>'\n"
                    f");"
                )
                query_whole = f"""
                    SELECT COUNT(DISTINCT TRIM(LOWER(email))) 
                    FROM ({clean_sql}) AS sub_whole
                    WHERE TRIM(LOWER(email)) NOT IN (
                        SELECT TRIM(LOWER(email)) 
                        FROM export_tracker 
                        WHERE TRIM(LOWER(project_number)) = :pn 
                        AND filters != :raw_sql
                    )
                """
            
            query_launched = """
                SELECT COUNT(DISTINCT TRIM(LOWER(email))) 
                FROM export_tracker 
                WHERE TRIM(LOWER(project_number)) = :pn AND filters = :raw_sql AND email IS NOT NULL
            """

            query_batches = """
                SELECT
                    DATE(datetimestamp)                      AS export_date,
                    COUNT(DISTINCT TRIM(LOWER(email)))       AS exported
                FROM export_tracker
                WHERE TRIM(LOWER(project_number)) = :pn
                AND filters = :raw_sql
                AND email IS NOT NULL
                GROUP BY DATE(datetimestamp)
                ORDER BY DATE(datetimestamp)
            """
            
            age_condition = extract_dob_age_condition(clean_sql)
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
                    AND et.filters = :raw_sql
                    AND et.email IS NOT NULL
                    AND ({drift_filter})
                """

            # Define localized thread workers targeting specific SQLAlchemy return shapes
            def fetch_scalar_worker(sql_str, params):
                with db.engine.connect() as conn:
                    return conn.execute(text(sql_str), params).scalar() or 0

            def fetch_dataframe_worker(sql_str, params):
                with db.engine.connect() as conn:
                    res = conn.execute(text(sql_str), params)
                    return pd.DataFrame(res.fetchall(), columns=res.keys())

            # Fire queries concurrently to the persistent connection pool
            future_whole = executor.submit(fetch_scalar_worker, query_whole, {"pn": pn_clean, "raw_sql": raw_sql})
            future_launched = executor.submit(fetch_scalar_worker, query_launched, {"pn": pn_clean, "raw_sql": raw_sql})
            future_batches = executor.submit(fetch_dataframe_worker, query_batches, {"pn": pn_clean, "raw_sql": raw_sql})
            
            future_drifted = None
            if query_drifted:
                future_drifted = executor.submit(fetch_scalar_worker, query_drifted, {"pn": pn_clean, "raw_sql": raw_sql})
                
            compiled_tasks.append({
                "raw_sql": raw_sql,
                "display_sql": display_sql,
                "has_date_drift": has_date_drift,
                "futures": {
                    "whole": future_whole,
                    "launched": future_launched,
                    "batches": future_batches,
                    "drifted": future_drifted
                }
            })

    # Safely gather threaded payloads as they finalize execution
    processed_audiences = []
    for task in compiled_tasks:
        whole_count = task["futures"]["whole"].result()
        launched_count = task["futures"]["launched"].result()
        batch_df = task["futures"]["batches"].result()
        drifted_count = task["futures"]["drifted"].result() if task["futures"]["drifted"] else 0
        
        raw_available = whole_count - launched_count
        available_count = max(0, raw_available)
        
        processed_audiences.append({
            "raw_sql": task["raw_sql"],
            "display_sql": task["display_sql"],
            "has_date_drift": task["has_date_drift"],
            "whole_count": whole_count,
            "launched_count": launched_count,
            "drifted_count": drifted_count,
            "batch_df": batch_df,
            "raw_available": raw_available,
            "available_count": available_count
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

            if data["raw_available"] < 0:
                st.warning(
                    f"⚠️ Pool has shrunk below launched count — "
                    f"{data['launched_count']:,} exported but only {data['whole_count']:,} "
                    "currently qualify. Investigate before launching further."
                )
            
            batch_df = data["batch_df"]
            if not batch_df.empty:
                batch_df = batch_df.rename(columns={
                    "export_date": "Export Date",
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
            else:
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
