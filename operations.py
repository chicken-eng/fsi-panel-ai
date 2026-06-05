import streamlit as st
import pandas as pd
from sqlalchemy import text
from fsi_ai import get_engine

# Define the popup/dialog box modal
@st.dialog("Project Action Console")
def open_action_popup(row_data):
    st.write(f"### Management Panel")
    st.markdown(f"**Target Project State:** `{row_data.get('project_state', 'N/A')}`")
    
    # Render full details in a clean key-value layout
    st.json(row_data)
    
    st.divider()
    st.info("⚡ This popup is ready to process customized operational actions.")
    
    # Blank placeholder for your future custom database query execution
    if st.button("Execute Action Query", type="primary", use_container_width=True):
        # Here you will write your connection and run_query code later
        st.success("Query placeholder executed successfully!")

def show_operations_page():
    st.title("Operations Activity Logs")
    
    # 1. Subheading requested
    st.subheader("Open projects")
    
    # 2. Reusing your existing database connection engine
    engine = get_engine()

    max_retries = 5
    delay = 3
    db_awake = False

    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                # Execute an ultra-lightweight ping query to force compute spin-up
                conn.execute(text("SELECT 1"))
                db_awake = True
                break
        except Exception as e:
            error_str = str(e).lower()
            is_conn_error = any(keyword in error_str for keyword in [
                "connection", "timeout", "closed", "ssl", "operationalerror"
            ])
            if is_conn_error and attempt < max_retries - 1:
                # Database is sleeping; wait and try again
                time.sleep(delay)
            else:
                break
                
    query = "SELECT project_number, project_name, project_type, topic, sharepoint_link, created_date FROM projects WHERE project_state = 'Open';"
    
    try:
        with engine.connect() as conn:
            # Safely pull data directly into a pandas dataframe
            df = pd.read_sql(text(query), conn)
        
        if not df.empty:
            st.caption("💡 Highlight any row below to trigger the operations action popup.")
            
            # hide_index=True eliminates the column of numbers entirely
            # on_select="rerun" turns the entire grid into clickable row links
            selection = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single_row"
            )
            
            # Catch when a user selects a specific table row
            if selection and selection.get("selection", {}).get("rows"):
                selected_row_idx = selection["selection"]["rows"][0]
                
                # Extract the exact row content into a clean dictionary map
                row_data = df.iloc[selected_row_idx].to_dict()
                
                # Launch the popup modal overlay passing the contextual details
                open_action_popup(row_data)
                
        else:
            st.info("There are currently no projects marked as 'Open'.")
            
    except Exception as e:
        st.error(f"Failed to load operational project metrics: {e}")
