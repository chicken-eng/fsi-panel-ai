import streamlit as st
import pandas as pd
from sqlalchemy import text
from fsi_ai import get_engine

def show_operations_page():
    st.title("Operations Activity Logs")
    
    # 1. Subheading requested
    st.subheader("Open projects")
    
    # 2. Reusing your existing database connection engine
    engine = get_engine()
    query = "SELECT * FROM projects WHERE project_state = 'Open';"
    
    try:
        with engine.connect() as conn:
            # Safely pull data directly into a pandas dataframe
            df = pd.read_sql(text(query), conn)
        
        # 3. Display the interactive table
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("There are currently no projects marked as 'Open'.")
            
    except Exception as e:
        st.error(f"Failed to load operational project metrics: {e}")
