import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableSequence
import time
import re
from collections import defaultdict

# ----------------------------
# Database connection
# ----------------------------
@st.cache_resource
def get_engine():
    url = (
        f"postgresql+psycopg2://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}"
        f"@{st.secrets['DB_HOST']}:{st.secrets.get('DB_PORT', 5432)}/{st.secrets['DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url)

# ----------------------------
# LLM Configuration
# ----------------------------
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )

SEMANTIC_GLOSSARY = """
SEMANTIC TRANSLATION GLOSSARY (Use this to map human terms to database values):
1. RESPONDENT TYPES (Filter via respondent_type_specification and respondent_type tables):
   - "Consumer"or "consumer" or "consumers" or "Consumers"  -> respondent_type.type_name = 'Consumer'
   - "HCP" or "hcp" or "Hcp" or "HCPs" or "hcps" or "HCP's" or "hcp's" or "Healthcare Professional" -> respondent_type.type_name = 'HCP'
   - "Patient/Caregiver" or "Patients" or "Patient" or "patient" or ""patients -> respondent_type.type_name = 'Patient/Caregiver'
   - "B2B" or "Business Respondent" -> respondent_type.type_name = 'B2B'

2. CALCULATING AGE:
   - "Age" or "Ages between X and Y" is NEVER a column. You must ALWAYS calculate it dynamically from r.date_of_birth using PostgreSQL AGE function.
   - Example for age 20-45: EXTRACT(YEAR FROM AGE(NOW(), r.date_of_birth)) BETWEEN 20 AND 45

3. HEALTHCARE PROFFESIONALS
   - Terms like "Physician", "Nurse", "Surgeon", "Pharmacist", "Technician", "Dentist", "Midwife", "Admin", "Manager", "Student", "Research/Laboratory", "Retired",
   are all HCP job titles (Filter via respondent_hcp_job_title and hcp_job_title tables). Any other values mentioned during a HCP search e.g. 'Oncology' is a specialty and must be filtered through respondent_hcp_specialty and hcp_job_specialty tables. 

4. DATES
   - When asked about when a respondent was created, made, first appeared etc filter using created_date from respondent_type_specification table. You will find multiple create_dates if respondent has different types, in this case take the oldest date.
   - When asked about when a respondent was last active filter via update_date from respondent_type_specification table. You will find multiple update_dates if respondent has different types, in this case take the newest date.
   - When asked about when a respondent was last active in relation to a project filter via last_activity_date from project_respondent table.

5. PROJECTS
   - Words like participate, took part, applied, will be used in relation to projects. When asked such question filter via project_respondent and projects tables.
   - If the question about a project requires a date or date range use the last_activity_date to filter.

5.ETHNICITY
  - Whenever asked about ethnicity filter via respondents table. If asked for whites use "White European" and "White Irish" and "White American" and "White British" and "Gypsy or Irish Traveller" and "Any other white background". 
"""

@st.cache_resource
def get_schema_description() -> str:
    """
    Pulls live schema from the database and formats it for the LLM"""
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            # 1. Get all tables in public schema
            tables_result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_type = 'BASE TABLE'
                  AND table_name NOT IN (
                    'staging_emails', 'staging_respondents', 
                    'staging_projects', 'staging_respondent_projects',
                    'error_log', 'survey_response'
                  )
                ORDER BY table_name
            """))
            tables = [row[0] for row in tables_result]
            
            # 2. Get columns + constraints for each table
            cols_result = conn.execute(text("""
                SELECT 
                    cols.table_name,
                    cols.column_name,
                    cols.data_type,
                    cols.is_nullable,
                    tc.constraint_type
                FROM information_schema.columns cols
                LEFT JOIN information_schema.key_column_usage kcu
                    ON cols.table_name = kcu.table_name 
                    AND cols.column_name = kcu.column_name
                    AND kcu.table_schema = 'public'
                LEFT JOIN information_schema.table_constraints tc
                    ON kcu.constraint_name = tc.constraint_name
                    AND tc.table_schema = 'public'
                WHERE cols.table_schema = 'public'
                  AND cols.table_name = ANY(:tables)
                ORDER BY cols.table_name, cols.ordinal_position
            """), {"tables": tables})
            
            # 3. Group columns by table
            table_cols = defaultdict(list)
            for row in cols_result:
                table_name, col_name, data_type, nullable, constraint = row
                # Format nicely: email (varchar, PK), first_name (varchar)
                col_str = f"{col_name} ({data_type}"
                if constraint == 'PRIMARY KEY':
                    col_str += ", PK"
                elif constraint == 'FOREIGN KEY':
                    col_str += ", FK"
                if nullable == 'NO':
                    col_str += ", NOT NULL"
                col_str += ")"
                table_cols[table_name].append(col_str)

            # 3. Get foreign key relationships
            fk_result = conn.execute(text("""
                SELECT
                    kcu.table_name AS from_table,
                    kcu.column_name AS from_col,
                    ccu.table_name AS to_table,
                    ccu.column_name AS to_col
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
                  AND kcu.table_name = ANY(:tables)
                ORDER BY kcu.table_name
            """), {"tables": tables})

            relationships = defaultdict(list)
            for row in fk_result:
                from_table, from_col, to_table, to_col = row
                relationships[from_table].append(
                    f"{from_col} → {to_table}.{to_col}"
                )
            
            # 4. Build a compact schema string
            schema_lines = [
                "We have a PostgreSQL database that contains the following tables:\n",
                "COLUMN INVENTORY (use these exact column names — do not invent columns):\n"
            ]

            for table in tables:
                cols = table_cols.get(table, [])
                schema_lines.append(f"TABLE: {table}")
                schema_lines.append(f"  Columns: {', '.join(cols)}")
                if relationships.get(table):
                    schema_lines.append(f"  Foreign keys: {', '.join(relationships[table])}")
                schema_lines.append("")
            
            schema_lines.append("RELATIONSHIPS SUMMARY:")
            schema_lines.append("Tables that have no declared FK but join via email: use r.email = other_table.email")
            schema_lines.append("")
            schema_lines.append(SEMANTIC_GLOSSARY)
            schema_lines.append("")
            schema_lines.append(BUSINESS_CONTEXT)
            
            return "\n".join(schema_lines)
            
    except Exception as e:
        # Fall back to static description if DB is unreachable
        st.warning(f"Could not load live schema, using static fallback: {e}")
        
        fallback_lines = [
            STATIC_SCHEMA_FALLBACK,
            SEMANTIC_GLOSSARY,
            "",
            BUSINESS_CONTEXT
        ]
        return "\n".join(fallback_lines)
        
# ----------------------------
# Schema context
# ----------------------------
BUSINESS_CONTEXT = """
IMPORTANT BUSINESS CONTEXT:
- respondent: Core table of the database. Nearly all table connect through this table.
- almost all table connect though a column called email. 
- unsubscribe_blacklist contains opted-out emails — ALWAYS exclude them

CRITICAL RULES YOU MUST ALWAYS FOLLOW:
1. ALWAYS exclude emails that appear in the unsubscribe_blacklist table from ANY query 
   result that returns respondent emails or counts unless specified otherwise. Always use:
   AND email NOT IN (SELECT email FROM unsubscribe_blacklist)
   or a LEFT JOIN with WHERE unsubscribe_blacklist.email IS NULL.
2. Always use lowercase table and column names.
3. Use PostgreSQL syntax only.
4. DISREGARD is_deleted and is_active columns from respondent and respondent_type_specification tables in your queries unless specified in the question.
5. Your response should only be a brief summary of what you found, never the raw rows themselves.
6. Several columns in the database are PostgreSQL enum types, not plain text. 
   These include but are not limited to: country, uk_region, county_state, gender, 
   ethnicity, relationship, job_status, job_title_tier, industry, 
   highest_education_level, annual_household_income, company_size, company_turnover, 
   years_in_business, approximate_salary_bracket, project_state, company_turnover.
   
   For ANY column that filters by a categorical or descriptive value, NEVER assume 
   the format or use abbreviations. Always use the full stored value exactly as it 
   appears in the database. For example: 'United States of America' not 'US', 
   'United Kingdom' not 'UK', 'Male' not 'M'.
   
   When unsure of the exact enum value, use ILIKE for partial matching instead:
   WHERE column::text ILIKE '%keyword%' This casts the enum to text first which avoids type errors entirely.
7. ALWAYS qualify every column name with its table alias when writing JOIN queries.
   Never write SELECT email, SELECT country etc when multiple tables are joined.
   Always write SELECT r.email, SELECT a.country etc.
   This applies to WHERE clauses, ON clauses, GROUP BY, and ORDER BY as well.
   Example: WHERE r.email NOT IN (...) not WHERE email NOT IN (...).
8.When a question asks to COUNT something, return a single aliased column.
    Example: SELECT COUNT(DISTINCT r.email) AS total_respondents
    Never return an unnamed count column.
9.When joining respondent to addresses, always use LEFT JOIN not INNER JOIN unless 
    the question specifically requires an address field to be present. Many respondents 
    may not have an address record and an INNER JOIN would silently exclude them from 
    counts.
10.Never use SELECT * in any query. Always specify the columns you need explicitly.
11. When filtering by date, always use TIMESTAMP WITH TIME ZONE safe comparisons (e.g., column_name >= '2024-01-01'::timestamptz). 
Use only date columns explicitly listed in the schema.
"""

STATIC_SCHEMA_FALLBACK = """
You are a data analyst assistant for a market research and panel management company.
The PostgreSQL database contains the following key tables:

- respondent: Core table of panel members. Fields include email (PK), first_name, last_name, 
  date_of_birth, phone_number, gender, ethnicity, is_deleted, ip_address.

- addresses: Respondent address info linked by email. Fields include country, uk_region, 
  county_state, city, postal_code.

- respondent_type_specification: What type a respondent is (consumer, HCP, patient, B2B etc), 
  linked by email and type_id. Includes is_active, respondent_tier.

- respondent_type: Lookup table for type names (type_id, type_name).

- conditions: Medical conditions lookup (condition_id, condition).

- respondent_condition_specification: Links respondents to their conditions via email and 
  condition_id. Includes professionally_diagnosed flag.

- hcp_job_title: Lookup for HCP job titles.
- hcp_job_specialty: Lookup for HCP specialties.
- hcp_level_of_expertise: Lookup for HCP expertise levels.
- respondent_hcp_job_title: Links respondents to HCP job titles.
- respondent_hcp_specialty: Links respondents to HCP specialties, includes hcp_number, 
  hcp_sub_specialty.
- respondent_hcp_level_of_expertise: Links respondents to expertise levels.

- socio_economic: Socio-economic profile per respondent linked by email. Fields include 
  job_status, current_job_title, job_title_tier, industry, highest_education_level, 
  annual_household_income.

- household: Household info per respondent linked by email. Fields include relationship, 
  children, number_of_children.

- clients: Client companies (client_id, client_name, created_date).

- projects: Research projects (project_number PK, project_name, client_id, topic, 
  project_type, project_state, created_date, end_date).

- project_respondent: Links respondents to projects via email and project_number. 
  Includes incentive, currency, interaction_level, last_activity_date.

- survey_response: Survey answers per respondent per project. Fields include email, 
  project_number, survey_question, survey_answer.

- mailings: Email campaigns (mailing_id, mailing_name).

- messages: Individual emails sent (system_message_id PK, mailing_id, subject, from_email).

- delivery_logs: Delivery status per message per recipient. Fields include system_message_id, 
  recipient_email, status, status_date, failure_type, failure_code, reason.

- engagement: Email engagement per message. Fields include system_message_id, first_open, 
  last_open, open_count, first_click, last_click, click_count, last_event_type.

- mx_records: MX provider info per domain.

- company_profile: B2B respondent company info linked by email. Fields include company_name, 
  company_size, company_turnover, years_in_business, industry, approximate_salary_bracket.

- providers / servers: Email sending infrastructure lookup tables.
"""

# ----------------------------
# Prompt templates
# ----------------------------

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert conversational context manager. Your sole job is to review a chat history between a data analyst and a user, look at the latest follow-up statement, and combine them into a single, self-contained, completely unambiguous question.

CRITICAL RULES:
1. Preserve Aggregation Context: If a previous question asked for a count ("how many", "total"), and the follow-up asks about a different category or subset (e.g., "and hcps?", "how about females?" "what about in the UK?"), you MUST carry over the count intent into the rewritten question. Do NOT change a count request into a list tracking request.
2. Maintain Existing Filters: If the conversation thread establishes baseline constraints (e.g., active users, specific years), keep those filters active in the rewritten question unless explicitly overridden by the follow-up.
3. Keep it brief: Output ONLY the completely rewritten standalone question. No markdown code fences, no commentary, no preamble.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

SIMPLE_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
{schema}

You are an expert PostgreSQL database engineer for a market research firm. 
Provide the highly optimized PostgreSQL query to answer the user's question.
Return ONLY the final SQL query enclosed strictly within ```sql and ``` markdown tags. 
Do not explain the SQL or provide any step-by-step reasoning.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

COMPLEX_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
{schema}

You are an expert PostgreSQL database engineer for a market research firm. 

Before writing the query, you MUST think through the problem step-by-step. Use this exact format:

**Reasoning:**
1. Identify the specific tables needed to answer the question.
2. Identify all filters and constraints requested by the user (e.g., age limits, gender, dates, respondent types).
3. Map those human terms to the exact database values using the SEMANTIC TRANSLATION GLOSSARY.
4. Plan the necessary JOINs.
5. Determine the output format: If the user asks "how many" or the question is a conversational follow-up to a previous COUNT, you MUST use COUNT. If they ask "who" or "show me", use a LIST.

**SQL:**
After your reasoning, provide the final, highly optimized PostgreSQL query enclosed strictly within ```sql and ``` markdown tags. Do not explain the SQL after writing it.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a data analyst reporting internal database results to a colleague.

Rules:
- Report ONLY what the data shows. Never compare to external benchmarks or real world statistics.
- If the result is a single value, respond in one short sentence stating just the number.
- Do not add commentary, caveats, or explanations unless the data is empty.
- When you run a SQL query that returns data, DO NOT generate a Markdown table of the results in your text response. Your response should only be a brief summary of what you found, never the raw rows themselves.
- If no data was returned, say: "No results were found for that question."
- If a question asked for a LIST of people or records, always provide r.email, r.first_name plus columns used/specified when filtering.
"""),
    ("human", """
A user asked: "{question}"

The query returned this data:
{data}
""")
])

VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a strict SQL code reviewer for a market research database.

Perform these two checks:
1. Filters: Does the SQL apply EVERY filter and condition requested in the user's question (e.g., age ranges, gender, respondent types)?
2. Output Format: Does the SQL use the correct aggregation? If the user asks "how many" or the question implies a count, the SQL MUST use COUNT(). If they ask for a list, it MUST select email, first_name, last_name.

- If the SQL accurately reflects all constraints AND the correct output format, reply EXACTLY with: VALID
- If the SQL is missing parameters or uses the wrong output format (e.g., returning a list when a count is expected), reply ONLY with a brief description of what is wrong. Example: "Missing filter for ages 20-45" or "Should be a COUNT() query, not a SELECT list."
"""),
    ("human", """
A user asked: "{question}"

The generated SQL is:
```sql
{sql}
""")
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
{rules}

You previously generated this SQL query:
{bad_sql}

HOWEVER, a code reviewer rejected this SQL for the following reason:
{missing_logic}

Rewrite the SQL query so that it properly includes the missing logic. 
Return ONLY the SQL query with no explanation, no markdown, no code fences.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# ----------------------------
# Core functions
# ----------------------------
def _parse_history(history) -> list:
    """Safely coerces standard session elements into strictly typed LangChain Message objects."""
    if not history:
        return []
    parsed = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "human"):
                parsed.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                parsed.append(AIMessage(content=content))
        elif isinstance(msg, tuple) and len(msg) == 2:
            role, content = msg
            if role in ("user", "human"):
                parsed.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                parsed.append(AIMessage(content=content))
        elif hasattr(msg, "content"):
            parsed.append(msg)
    return parsed

def extract_sql_from_cot(llm_response: str) -> str:
    """
    Extracts the SQL query from the LLM's response, ignoring the reasoning.
    Looks specifically for markdown SQL blocks.
    """
    # Primary match: look for ```sql [query] ```
    match = re.search(r"```sql\n(.*?)\n```", llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback match: in case the LLM forgets the 'sql' tag and just uses ```
    match_fallback = re.search(r"```(.*?)```", llm_response, re.DOTALL)
    if match_fallback:
        return match_fallback.group(1).strip()
        
    # Failsafe: If no markdown is found, return the raw response 
    # (Though with our prompt, this should rarely happen)
    return llm_response.strip()

def validate_sql_intent(question: str, sql: str) -> str:
    """Checks if the generated SQL dropped any user parameters."""
    llm = get_llm()
    chain = VALIDATION_PROMPT | llm
    result = chain.invoke({"question": question, "sql": sql}).content.strip()
    return result

def is_complex_query(question: str) -> bool:
    """
    Evaluates the user's question to determine if it requires Chain-of-Thought reasoning.
    Zero-token heuristic based on word count and complexity keywords.
    """
    q_lower = question.lower()
    
    # Keywords that imply multi-table joins, date filtering, or complex aggregations
    complexity_indicators = [
        "between", "average", "ratio", "compare", "trend", "month", "year",
        "both", "multiple", "except", "without", "versus", "vs", "most", 
        "least", "top", "bottom", "percentage", "group by"
    ]
    
    # Route to CoT if the question is heavily constrained (long)
    if len(q_lower.split()) > 15:
        return True
        
    # Route to CoT if it hits any complexity keywords
    if any(word in q_lower for word in complexity_indicators):
        return True
        
    return False

def run_query(sql: str, max_retries: int = 5, delay: int = 3) -> pd.DataFrame | None:
    """Runs the query with a retry loop to handle Neon's cold starts."""
    engine = get_engine()
    df = None  # Initialize df here
    
    # st.status provides a spinner UI that we can update text for dynamically
    with st.status("Connecting to database...", expanded=True) as status:
        for attempt in range(max_retries):
            try:
                with engine.connect() as conn:
                    status.update(label="Executing query...", state="running")
                    result = conn.execute(text(sql))
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    status.update(label="Query successful!", state="complete", expanded=False)
                    break  # Exit the retry loop gracefully instead of returning early
                    
            except Exception as e:
                error_str = str(e).lower()
                # Check if the error is likely due to the database being asleep
                is_conn_error = any(keyword in error_str for keyword in [
                    "connection", "timeout", "closed", "ssl", "operationalerror"
                ])
                
                if is_conn_error and attempt < max_retries - 1:
                    status.update(
                        label=f"Waking up database... (Attempt {attempt + 1} of {max_retries})", 
                        state="running"
                    )
                    time.sleep(delay)  # Wait a few seconds before trying again
                else:
                    # If it's a strict SQL syntax error or we ran out of retries, fail out
                    status.update(label="Query failed.", state="error")
                    st.session_state["last_sql_error"] = str(e)
                    st.error(f"SQL execution error: {e}")
                    break  # Also break here on complete failure
                    
    # Return df OUTSIDE the context manager so Streamlit fully closes the UI status
    return df

def get_column_samples(sql: str) -> str:
    """Looks at the SQL, finds the tables used, and fetches distinct values for text columns."""
    engine = get_engine()
    samples = []

    words = sql.lower().split()
    tables = []
    for i, word in enumerate(words):
        if word in ("from", "join") and i + 1 < len(words):
            table = words[i + 1].strip("(),;")
            if table and not table.startswith("("):
                tables.append(table)

    try:
        with engine.connect() as conn:
            for table in set(tables):
                try:
                    col_result = conn.execute(text(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}' 
                        AND data_type IN ('text', 'character varying', 'USER-DEFINED')
                        LIMIT 8
                    """))
                    columns = [row[0] for row in col_result]

                    for col in columns:
                        try:
                            val_result = conn.execute(text(f"""
                                SELECT DISTINCT {col} 
                                FROM {table} 
                                WHERE {col} IS NOT NULL 
                                LIMIT 5
                            """))
                            values = [str(row[0]) for row in val_result]
                            if values:
                                samples.append(f"{table}.{col}: {', '.join(values)}")
                        except:
                            pass
                except:
                    pass
    except:
        pass

    return "\n".join(samples)

def get_real_columns_for_sql(sql: str) -> str:
    """
    Given a SQL string, extracts all table names referenced and returns
    their real column lists from information_schema. Used for UndefinedColumn retries.
    """
    engine = get_engine()
    lines = []
    
    # Extract table names from FROM and JOIN clauses
    words = sql.lower().split()
    tables = []
    for i, word in enumerate(words):
        if word in ("from", "join") and i + 1 < len(words):
            candidate = words[i + 1].strip("(),;")
            if candidate and not candidate.startswith("(") and not candidate.startswith("select"):
                tables.append(candidate)
    
    if not tables:
        return ""
    
    try:
        with engine.connect() as conn:
            for table in set(tables):
                try:
                    result = conn.execute(text("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = :table
                        ORDER BY ordinal_position
                    """), {"table": table})
                    cols = [f"{row[0]} ({row[1]})" for row in result]
                    if cols:
                        lines.append(f"Table '{table}' has these columns: {', '.join(cols)}")
                except Exception:
                    pass
    except Exception:
        pass
    
    return "\n".join(lines)

# ----------------------------
# Execution Pipeline
# ----------------------------

def generate_sql_with_retry(question: str, history: list = None) -> tuple[str | None, pd.DataFrame | None]:
    """Generates SQL, validates intent, runs it, and handles DB retries."""

    parsed_history = _parse_history(history)
    
    with st.expander("🔍 Query Process", expanded=True):
        # Step 0: Context Condensation Layer
        st.markdown("**Step 0: Synchronizing Context ...**")
        if parsed_history:
            llm = get_llm()
            context_chain = CONTEXTUALIZE_PROMPT | llm
            condensed_question = context_chain.invoke({
                "history": parsed_history,
                "question": question
            }).content.strip()

            if condensed_question != question:
                st.info(f"🔄 Rephrased contextually to: *\"{condensed_question}\"*")
                question = condensed_question
        else:
            st.info("🆕 Fresh chat thread started.")
    
        # Step 1
        st.markdown("**Step 1: Generating SQL...**")
        sql = generate_sql(question, history=history)
        
        if not sql:
            st.error("Could not generate a valid SQL query.")
            return None, None
            
        if not sql.lower().startswith("select"):
            st.info("No database query required for this message.")
            # We return the conversational text as 'sql', but df as None.
            return sql, None
        
        st.code(sql, language="sql")

        # Step 2: Validate Intent
        st.markdown("**Step 2: Validating Query Intent...**")
        validation_result = validate_sql_intent(question, sql)
        
        if validation_result != "VALID":
            st.warning(f"⚠️ Code Reviewer flagged an issue: {validation_result}")
            st.markdown("**Step 2.5: Rewriting SQL to include missing parameters...**")
            
            # Run the targeted rewrite
            llm = get_llm()
            chain = REWRITE_PROMPT | llm
            sql = chain.invoke({
                "rules": f"{SEMANTIC_GLOSSARY}\n\n{BUSINESS_CONTEXT}",
                "history": parsed_history,
                "question": question,
                "bad_sql": sql,
                "missing_logic": validation_result
            }).content.strip()
            
            st.code(sql, language="sql")
        else:
            st.info("✅ Query passed intent validation.")
        
        # Step 3
        st.markdown("**Step 3: Running query...**")
        df = run_query(sql)
        sql_error = st.session_state.pop("last_sql_error", None)
        
        if df is not None and not df.empty:
            st.success(f"Query returned {len(df)} row(s). No retry needed.")
            return sql, df
        
        # Step 4 — retry
        if (df is not None and df.empty) or (df is None and sql_error):
            
            if sql_error and "undefinedcolumn" in sql_error.lower():
                st.warning("⚠️ Query used a column that doesn't exist. Fetching real column names...")
                st.markdown("**Step 4: Injecting real column inventory...**")
            
                # Extract the offending table from the error or from the SQL
                real_columns = get_real_columns_for_sql(sql)

                if real_columns:
                    llm = get_llm()
                    retry_prompt = ChatPromptTemplate.from_messages([
                        ("system", """
{rules}

You previously generated this SQL query:
{bad_sql}

It failed with this error:
{error}

Here are the ACTUAL columns that exist on the tables used in your query:
{real_columns}

IMPORTANT: Only use column names from the list above. Do not invent or assume any column names.
Rewrite the SQL query using only the real columns listed.
Return ONLY the SQL query with no explanation, no markdown, no code fences.
"""),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}")
                    ])
                    chain = retry_prompt | llm
                    sql = chain.invoke({
                        "rules": f"{SEMANTIC_GLOSSARY}\n\n{BUSINESS_CONTEXT}",
                        "history": parsed_history,
                        "question": question,
                        "bad_sql": sql,
                        "real_columns": real_columns,
                        "error": sql_error
                    }).content.strip()
                
                    st.markdown("**Step 4: Retried SQL:**")
                    st.code(sql, language="sql")
                    
                    df = run_query(sql)
                    sql_error = st.session_state.pop("last_sql_error", None)
                    if df is not None and not df.empty:
                        st.success(f"✅ Retry successful — {len(df)} row(s).")
                    elif sql_error:
                        st.error(f"Retry also failed: {sql_error}")
                    else:
                        st.info("Retry ran but returned 0 results.")
                    return sql, df

            # --- Branch B: Zero results — inject real enum values ---
            else:
                if sql_error:
                    st.warning(f"⚠️ Query failed: {sql_error}")
                else:
                    st.warning("⚠️ Query returned 0 results. Checking actual stored values...")
                
                st.markdown("**Step 3: Fetching column value samples...**")
                samples = get_column_samples(sql)
                
                if samples:
                    # Don't dump raw samples to UI — just confirm we got them
                    st.info("✅ Retrieved column value samples for retry.")
                    st.markdown("**Step 4: Retrying with correct values...**")
                    
                    llm = get_llm()
                    retry_prompt = ChatPromptTemplate.from_messages([
                        ("system", """
{rules}

You previously generated this SQL query:
{bad_sql}

It returned zero results. Here are the actual distinct values stored in the relevant columns:
{samples}

Using these exact values, rewrite the SQL query.
Return ONLY the SQL query with no explanation, no markdown, no code fences.
"""),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}")
                    ])
                    chain = retry_prompt | llm
                    sql = chain.invoke({
                        "rules": f"{SEMANTIC_GLOSSARY}\n\n{BUSINESS_CONTEXT}",
                        "history": parsed_history,
                        "question": question,
                        "bad_sql": sql,
                        "samples": samples
                    }).content.strip()
                    
                    st.markdown("**Step 4: Retried SQL:**")
                    st.code(sql, language="sql")
                    
                    df = run_query(sql)
                    sql_error = st.session_state.pop("last_sql_error", None)
                    if df is not None and not df.empty:
                        st.success(f"✅ Retry successful — {len(df)} row(s).")
                    elif sql_error:
                        st.error(f"Retry also failed: {sql_error}")
                    else:
                        st.info("Retry ran but returned 0 results.")
                else:
                    st.warning("Could not fetch column samples for retry.")
    
    return sql, df

def generate_sql(question: str, history: list = None) -> str | None:
    """Generates SQL using CoT reasoning and structured chat history."""
    if history is None:
        history = []
    
    llm = get_llm()
    
    # 1. Evaluate complexity
    requires_cot = is_complex_query(question)
    
    # 2. Route to the correct prompt
    prompt_template = COMPLEX_SQL_PROMPT if requires_cot else SIMPLE_SQL_PROMPT
    chain = prompt_template | llm

    try:  
        raw_response = chain.invoke({
            "schema": get_schema_description(),
            "history": history,
            "question": question
        }).content
        
        # Strip out the reasoning, leaving only the executable SQL
        clean_sql = extract_sql_from_cot(raw_response)
        return clean_sql
        
    except Exception as e:
        st.error(f"Error generating SQL: {e}")
        return None

def generate_response(question: str, df: pd.DataFrame) -> str | None:
    llm = get_llm()
    chain = RESPONSE_PROMPT | llm
    data_str = df.to_string(index=False) if df is not None else "No data returned."
    return chain.invoke({"question": question, "data": data_str}).content

# ----------------------------
# Interface Compatibility Stubs
# ----------------------------
def generate_questions_cached():
    return [
        "How many active respondents do we have?",
        "Which clients have the most projects?",
        "How many respondents are healthcare professionals?",
        "What are the top 5 conditions in our panel?",
        "Show me all projects currently in progress",
        "How many respondents have unsubscribed?",
    ]

def generate_sql_cached(question: str, history: list = None):
    return generate_sql_with_retry(question, history=history)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    return sql is not None and sql.strip().lower().startswith("select")

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    return run_query(sql)

def should_generate_chart_cached(question, sql, df):
    return False  # charts dropped for prototype

def generate_plotly_code_cached(question, sql, df):
    return None

def generate_plot_cached(code, df):
    return None

def generate_followup_cached(question, sql, df):
    return []

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    if df is None or df.empty:
        return "No data was returned for that question."

    if len(df) == 1:
        return generate_response(question, df)
    elif len(df) <= 5:
        return generate_response(question, df.head(5))
    else:
        # For large result sets, just confirm what was returned
        cols = ", ".join(df.columns.tolist())
        return f"Query returned {len(df)} records with columns: {cols}."
