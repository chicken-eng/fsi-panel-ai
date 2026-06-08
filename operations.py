import re
import time
import pandas as pd
import streamlit as st
from sqlalchemy import text
from fsi_ai import get_engine

def get_all_project_sqls(engine, project_number):
    """Fetches ALL distinct saved AI queries for this project from the export tracker."""
    query = "SELECT DISTINCT filters FROM export_tracker WHERE project_number = :pn AND filters IS NOT NULL"
    try:
        with engine.connect() as conn:
            # Fetch all distinct sql strings and return them as a list
            result = conn.execute(text(query), {"pn": str(project_number).lower()})
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

@st.dialog("Project Action Console")
def open_action_popup(row_data):
    st.write(f"### Management Panel")
    
    project_number = row_data.get('project_number', 'Unknown')
    st.markdown(f"**Target Project:** `{project_number}`")
    
    st.divider()
    
    engine = get_engine()
    
    with st.spinner("Calculating live sample metrics..."):
        base_sqls = get_all_project_sqls(engine, project_number)
        
        if not base_sqls:
            st.info("No saved filters/exports found for this project yet. Launch a sample from the FSI AI page to populate these metrics.")
        else:
            # Loop through every distinct target audience query found for this project
            for i, raw_sql in enumerate(base_sqls):
                
                # If there is more than one, give them clean subheadings so the UI doesn't blur together
                if len(base_sqls) > 1:
                    st.markdown(f"#### Target Audience Definition #{i + 1}")
                    
                clean_sql = clean_sql_for_counts(raw_sql)
                
                # 1. Whole Sample: Count everything returned by this specific AI query
                query_whole = f"SELECT COUNT(DISTINCT TRIM(LOWER(email))) FROM ({clean_sql}) AS sub_whole"
                
                # 2. Launched Sample: Count emails exported under THIS SPECIFIC query string
                query_launched = """
                    SELECT COUNT(DISTINCT TRIM(LOWER(email))) 
                    FROM export_tracker 
                    WHERE TRIM(LOWER(project_number)) = :pn AND filters = :raw_sql
                """
                # 4. Drifted: exported emails that no longer appear in the current SQL results
                query_drifted = f"""
                    SELECT COUNT(DISTINCT TRIM(LOWER(et.email)))
                    FROM export_tracker et
                    WHERE TRIM(LOWER(et.project_number)) = :pn
                    AND et.filters = :raw_sql
                    AND et.email IS NOT NULL
                    AND TRIM(LOWER(et.email)) NOT IN (
                        SELECT TRIM(LOWER(sub.email)) 
                        FROM ({clean_sql}) AS sub
                        WHERE sub.email IS NOT NULL
                   )
                """
                # Execute the queries for this specific loop iteration
                try:
                    with engine.connect() as conn:
                        pn_clean = str(project_number).lower().strip()
                        whole_count = conn.execute(text(query_whole)).scalar() or 0
                        launched_count = conn.execute(text(query_launched), {
                            "pn": pn_clean,
                            "raw_sql": raw_sql
                        }).scalar() or 0
                        drifted_count  = conn.execute(text(query_drifted), {
                            "pn": pn_clean,
                            "raw_sql": raw_sql
                        }).scalar() or 0

                    available_count = whole_count - launched_count

                    if available_count < 0:
                       st.warning(
                           f"⚠️ Pool has shrunk: {launched_count:,} exported "
                           f"but only {whole_count:,} currently qualify. "
                           f"Investigate before launching further."
                       )
                       available_count = 0
                    else:
                       pass
                    
                    # Render the metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Whole Sample", f"{whole_count:,}")
                    m2.metric("Launched Sample", f"{launched_count:,}")
                    m3.metric("Available Sample", f"{available_count:,}")
                    m4.metric("Drifted Out",      f"{drifted_count:,}",            # ← new
                               delta=f"-{drifted_count:,}" if drifted_count > 0 else None,
                               delta_color="inverse")
                    
                    # Expandable code block for inspection
                    with st.expander("View Base Target Parameters (SQL)"):
                        st.code(clean_sql, language="sql")
                        
                except Exception as e:
                    st.error("Failed to execute sample calculations against the database.")
                    with st.expander("View Error Details"):
                        st.exception(e)
                
                # Add a divider between multiple queries for visual cleanliness, unless it's the last one
                if i < len(base_sqls) - 1:
                    st.divider()

    st.divider()
    st.info("⚡ This popup is ready to process customized operational actions.")
    
    if st.button("Execute Action Query", type="primary", use_container_width=True):
        st.success("Query placeholder executed successfully!")

def show_operations_page():
    st.title("Operations Activity Logs")
    st.subheader("Open projects")
    
    engine = get_engine()
    query = "SELECT project_number, project_name, project_type, topic, sharepoint_link, created_date FROM projects WHERE project_state = 'Open' order by project_type;"
    
    # ------------------------------------------------------------
    # NEON SERVERLESS WAKEUP BUFFER
    # ------------------------------------------------------------
    max_retries = 5
    delay = 3
    db_awake = False
    
    with st.spinner("Synchronizing connection with database..."):
        for attempt in range(max_retries):
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    db_awake = True
                    break
            except Exception as e:
                error_str = str(e).lower()
                is_conn_error = any(keyword in error_str for keyword in [
                    "connection", "timeout", "closed", "ssl", "operationalerror"
                ])
                if is_conn_error and attempt < max_retries - 1:
                    st.caption(f"⏳ Neon compute node is waking up... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    break

    if not db_awake:
        st.error("❌ The database connection timed out while waking up. Please refresh the page to retry.")
        return

    # ------------------------------------------------------------
    # DATA RETRIEVAL & LAYOUT
    # ------------------------------------------------------------
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
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
