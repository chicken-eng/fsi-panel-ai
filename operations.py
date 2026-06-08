import re
import pandas as pd
import streamlit as st
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

@st.dialog("Project Action Console")
def open_action_popup(row_data):
    st.write(f"### Management Panel")
    
    project_number = row_data.get('project_number', 'Unknown')
    st.markdown(f"**Target Project:** `{project_number}`")
    
    st.divider()
    
    db = get_db()
    
    with st.spinner("Calculating live sample metrics..."):
        base_sqls = get_all_project_sqls(project_number)
        
        if not base_sqls:
            st.info("No saved filters/exports found for this project yet. Launch a sample from the FSI AI page to populate these metrics.")
        else:
            # Loop through every distinct target audience query found for this project
            for i, raw_sql in enumerate(base_sqls):
                
                # If there is more than one, give them clean subheadings so the UI doesn't blur together
                if len(base_sqls) > 1:
                    st.markdown(f"#### Target Audience Definition #{i + 1}")
                    
                clean_sql = clean_sql_for_counts(raw_sql)
                pn_clean  = str(project_number).lower().strip()

                if i == 0:
                    # First target audience: Just show the pure demographic SQL
                    display_sql = clean_sql
                else:
                    # Subsequent audiences: Wrap it to show the visual exclusion logic
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

                # ── Detect which date conditions exist in this SQL ──────────
                age_condition    = extract_dob_age_condition(clean_sql)
                update_condition = extract_update_date_condition(clean_sql)
                drift_parts = []
                if age_condition:
                    drift_parts.append(f"NOT ({age_condition})")
                if update_condition:
                    drift_parts.append(f"NOT ({update_condition})")
                has_date_drift = bool(drift_parts)
                
                # 1. Whole Sample: Count everything returned by this specific AI query
                if i == 0:
                    # Target Audience #1: Show the true grand total demographic pool (no exclusions)
                    query_whole = f"SELECT COUNT(DISTINCT TRIM(LOWER(email))) FROM ({clean_sql}) AS sub_whole"
                else:
                    # Target Audience #2+: Show available pool MINUS anyone claimed by previous project queries
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
                
                # 2. Launched Sample: Count emails exported under THIS SPECIFIC query string
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
                if has_date_drift:
                    # Only checks date conditions — NOT location, ethnicity, type etc.
                    drift_filter  = " OR ".join(drift_parts)
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
                # Execute the queries for this specific loop iteration
                try:
                    with engine.connect() as conn:
                        pn_clean = str(project_number).lower().strip()
                        
                        whole_count = db.execute(query_whole, {
                            "pn": pn_clean,
                            "raw_sql": raw_sql
                        }).scalar() or 0
                        
                        launched_count = db.execute(query_launched, {
                            "pn": pn_clean,
                            "raw_sql": raw_sql
                        }).scalar() or 0

                        batch_result = db.execute(
                            query_batches, {"pn": pn_clean, "raw_sql": raw_sql}
                        )
                        batch_df = pd.DataFrame(
                            batch_result.fetchall(), columns=batch_result.keys()
                        )
                        
                        drifted_count = 0
                        if has_date_drift:
                            drifted_count = conn.execute(
                                text(query_drifted), {"pn": pn_clean, "raw_sql": raw_sql}
                            ).scalar() or 0

                    
                    raw_available   = whole_count - launched_count
                    available_count = max(0, raw_available)

                    if has_date_drift:
                        m1, m2, m3, m4 = st.columns(4)
                        m4.metric(
                            "Drifted Out",
                            f"{drifted_count:,}",
                            delta=f"-{drifted_count:,}" if drifted_count > 0 else None,
                            delta_color="inverse",
                            help="Exported respondents who no longer satisfy the age "
                                 "or date-based criteria in the original query."
                        )
                    else:
                        m1, m2, m3 = st.columns(3)
                        
                    m1.metric("Whole Sample", f"{whole_count:,}")
                    m2.metric("Launched Sample", f"{launched_count:,}")
                    m3.metric("Available Sample", f"{available_count:,}")

                    # Warn if pool has shrunk below what was already exported
                    if raw_available < 0:
                        st.warning(
                            f"⚠️ Pool has shrunk below launched count — "
                            f"{launched_count:,} exported but only {whole_count:,} "
                            "currently qualify. Investigate before launching further."
                        )
                    
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
                        
                        # Loop through every row/batch item to isolate them vertically
                        for idx, row in batch_df.iterrows():
                            # Clean stacked metric row representation
                            st.markdown(f"📅 **Export Date:** `{row['Export Date']}`   |   ✉️ **Emails Exported:** `{row['Emails Exported']:,}`")
                            
                            # Places the SQL expander directly below this specific row 
                            with st.expander("View Base Target Parameters (SQL)", expanded=False):
                                st.code(display_sql, language="sql")
                    else:
                        # No batch data yet — just show SQL
                        with st.expander("View Base Target Parameters (SQL)", expanded=False):
                            st.code(display_sql, language="sql")

                except Exception as e:
                    st.error("Failed to execute sample calculations against the database.")
                    with st.expander("View Error Details"):
                        st.exception(e)

                if i < len(base_sqls) - 1:
                    st.divider()

def show_operations_page():
    st.title("Operations Activity Logs")
    st.subheader("Open projects")
    
    db = get_db()
    query = "SELECT project_number, project_name, project_type, topic, sharepoint_link, created_date FROM projects WHERE project_state = 'Open' order by project_type;"
    
    # ------------------------------------------------------------
    # DATA RETRIEVAL & LAYOUT
    # ------------------------------------------------------------
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
