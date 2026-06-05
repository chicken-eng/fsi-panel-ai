import re
import time
import pandas as pd
import streamlit as st
from sqlalchemy import text
from fsi_ai import get_engine

def get_project_sql(engine, project_number):
    """Fetches the saved AI query for this project from the export tracker."""
    query = "SELECT filters FROM export_tracker WHERE project_number = :pn LIMIT 1"
    try:
        with engine.connect() as conn:
            # .scalar() returns the first column of the first row (the SQL string)
            return conn.execute(text(query), {"pn": str(project_number).lower()}).scalar()
    except Exception as e:
        st.warning(f"Could not retrieve base SQL: {e}")
        return None

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
    state = row_data.get('state', 'Unknown')
    st.markdown(f"**Target Project:** `{project_number}` | **State:** `{row_data.get('project_state', '{state}')}`")
    
    st.divider()
    
    engine = get_engine()
    
    with st.spinner("Calculating live sample metrics..."):
        base_sql = get_project_sql(engine, project_number)
        
        if not base_sql:
            st.info("No saved filters/exports found for this project yet. Launch a sample from the FSI AI page to populate these metrics.")
        else:
            # 1. Clean the saved query
            clean_sql = clean_sql_for_counts(base_sql)
            
            # 2. Define the three exact counting queries
            # Whole Sample: Count everything returned by the clean AI query
            query_whole = f"SELECT COUNT(*) FROM ({clean_sql}) AS sub_whole"
            
            # Launched Sample: Count unique emails explicitly saved in the tracker for this project
            query_launched = "SELECT COUNT(DISTINCT email) FROM export_tracker WHERE project_number = :pn"
            
            # Available Sample: Count everything from the clean AI query that is NOT in the tracker
            query_available = f"""
                SELECT COUNT(*) FROM ({clean_sql}) AS sub_avail
                WHERE sub_avail.email NOT IN (
                    SELECT email FROM export_tracker WHERE project_number = '{str(project_number).lower()}'
                )
            """
            
            # 3. Execute queries
            try:
                with engine.connect() as conn:
                    whole_count = conn.execute(text(query_whole)).scalar() or 0
                    launched_count = conn.execute(text(query_launched), {"pn": str(project_number).lower()}).scalar() or 0
                    available_count = conn.execute(text(query_available)).scalar() or 0
                    
                # 4. Render the metrics side-by-side
                m1, m2, m3 = st.columns(3)
                m1.metric("Whole Sample", f"{whole_count:,}")
                m2.metric("Launched Sample", f"{launched_count:,}")
                m3.metric("Available Sample", f"{available_count:,}")
                
                # Optional: Let you inspect the cleaned SQL it ran for debugging
                with st.expander("View Base Target Parameters (SQL)"):
                    st.code(clean_sql, language="sql")
                    
            except Exception as e:
                st.error("Failed to execute sample calculations against the database.")
                st.exception(e)

def show_operations_page():
    st.title("Operations Activity Logs")
    st.subheader("Open projects")
    
    engine = get_engine()
    query = "SELECT project_number, project_name, project_type, topic, sharepoint_link, created_date FROM projects WHERE project_state = 'Open';"
    
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
