import time
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

class DatabaseManager:
    def __init__(self):
        """Initializes the connection engine with an active connection pool."""
        # Pull configurations from Streamlit secrets
        db_user = st.secrets['DB_USER']
        db_password = st.secrets['DB_PASSWORD']
        db_host = st.secrets['DB_HOST']
        db_port = st.secrets.get('DB_PORT', 5432)
        db_name = st.secrets['DB_NAME']
        
        url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
        
        # Performance tuning parameters added to keep connections warm and optimized
        self.engine = create_engine(
            url,
            pool_size=10,          # Keeps up to 10 persistent connections open
            max_overflow=5,        # Allows up to 5 additional burst connections
            pool_recycle=1800,     # Automatically recycles connections older than 30 mins
            pool_pre_ping=True     # Pessimistic disconnect handling (tests connection before using)
        )
        self.is_woken = False

    def ensure_awake(self, max_retries=5, delay=3):
        """Wakes up the Neon serverless compute node if it is asleep."""
        if self.is_woken:
            return True
            
        for attempt in range(max_retries):
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    self.is_woken = True
                    return True
            except Exception as e:
                error_str = str(e).lower()
                is_conn_error = any(kw in error_str for kw in ["connection", "timeout", "closed", "ssl", "operationalerror"])
                
                if is_conn_error and attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise e
        return False

    def get_df(self, query: str, params: dict = None) -> pd.DataFrame:
        """Helper method to execute a query and return a clean pandas DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)

    def execute(self, query: str, params: dict = None):
        """Helper method to execute single scalar values or insert/update operations."""
        with self.engine.connect() as conn:
            return conn.execute(text(query), params or {})


@st.cache_resource(show_spinner=False)
def get_db():
    """
    Acts as our persistent global connection gateway.
    Streamlit caches this specific object in memory across ALL script reruns 
    and user sessions, preserving the internal connection pool.
    """
    return DatabaseManager()
