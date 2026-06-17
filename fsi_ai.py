import streamlit as st
import pandas as pd
from sqlalchemy import text  # No longer need create_engine directly here
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableSequence
import time
import re
import hashlib
from collections import defaultdict

# ─── NEW: IMPORT OUR CENTRALIZED DATABASE MANAGER ───
from database import get_db

# ----------------------------
# LLM Configuration
# ----------------------------

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )


SEMANTIC_GLOSSARY = """
SEMANTIC TRANSLATION GLOSSARY (Mappings for human terms to database elements):
1. RESPONDENT TYPES (Scoping via respondent_type_specification and respondent_type tables):
   - Variations of "Consumer" -> respondent_type.type_name = 'Consumer'
   - Variations of "HCP" -> respondent_type.type_name = 'HCP'
   - Variations of "Patient" or "Caregiver" -> respondent_type.type_name = 'Patient/Caregiver'
   - Variations of "B2B" or "Business Respondent" -> respondent_type.type_name = 'B2B'

2. DYNAMIC AGE DERIVATION:
   - Querying for age or age ranges requires dynamic calculation from the date of birth field using the PostgreSQL AGE function.
   - Execution Template for an age constraint (e.g., ages 20 to 45): EXTRACT(YEAR FROM AGE(NOW(), r.date_of_birth)) BETWEEN 20 AND 45

3. HEALTHCARE PROFESSIONALS:
   - Map terms like "Physician", "Nurse", "Surgeon", "Pharmacist", "Technician", "Dentist", "Midwife", "Admin", "Manager", "Student", "Research/Laboratory", "Retired" to HCP job titles using the respondent_hcp_job_title and hcp_job_title tables.
   - Map clinical sub-domains, specialties, or medical topics (e.g., 'Oncology') to HCP specialties using the respondent_hcp_specialty and hcp_job_specialty tables.

4. TIMESTAMPS & LIFE CYCLE DATES:
   - Baseline creation, registration, or initial appearance queries: Filter using the `created_date` column in the `respondent_type_specification` table. Grouped entities with multiple type associations must resolve to the oldest baseline date via MIN().
   - Last active or interaction queries: Filter using the `update_date` column in the `respondent_type_specification` table. Grouped entities with multiple type associations must resolve to the newest interaction date via MAX().
   - Specific project interaction timelines: Filter using the `last_activity_date` column in the `project_respondent` table.

5. PROJECT PARTICIPATION:
   - Operational terms like "participate", "took part", or "applied" signify explicit project histories. Filter these requests through the `project_respondent` and `projects` tables.
   - Apply project-specific time boundaries to the `last_activity_date` column.

6. ETHNICITY MAPPING:
   - Querying for ethnicity requirements filters directly against the `respondents` table.
   - "Whites" or "Caucasians" matching sequence: WHERE r.ethnicity IN ('White European', 'White Irish', 'White American', 'White British', 'Gypsy or Irish Traveller', 'Any other white background').

7. ENUM COMPLIANCE:
   - Several attributes utilize PostgreSQL enum types (e.g., country, uk_region, county_state, gender, industry, job_status).
   - Target full, literal stored string values explicitly (e.g., 'United States of America', 'United Kingdom', 'Male'). 
   - If the exact stored string representation is ambiguous, convert the enum type to text inline for safe partial string comparison: WHERE column::text ILIKE '%keyword%'.

8. GEOGRAPHIC DEMOGRAPHICS:
   - Target city filters using partial text matching via ILIKE to ensure regional capture.
"""

# ----------------------------
# Export Tracking Configuration
# ----------------------------
EXPORT_KEYWORD_PATTERN = re.compile(r'\bexport(s|ing|ed)?\b', re.IGNORECASE)
PROJECT_NUMBER_PATTERN = re.compile(r'\bfsi[a-z0-9]{4,}\b',  re.IGNORECASE)

EXPORT_OVERRIDE_PHRASES = [
    "disregard previous export",
    "ignore previous export",
    "include everyone even if already exported",
    "include all even if exported",
    "include previously exported",
    "reset tracking for project",
    "re-export",
    "override export",
    "export all including previous",
]


@st.cache_resource
def get_schema_description() -> str:
    """Pulls live schema from the database and formats it for the LLM"""
    db = get_db()
    db_awake = False

    # ⚡ REPLACED MESSY MANUAL LOOP WITH CENTRALIZED WAKEUP ROUTINE
    try:
        db_awake = db.ensure_awake()
    except Exception:
        db_awake = False

    def build_structured_payload(catalog_content: str) -> str:
        return f"""# SYSTEM DATA MANUAL & OPERATIONAL RULES

## SECTION 1: DATABASE ENVIRONMENT PHYSICAL DICTIONARY
[CRITICAL] You must build queries utilizing ONLY the physical table names and specific columns inventoried inside this block. Do not extrapolate, assume, or invent schema configurations.

<database_schema_inventory>
{catalog_content}
</database_schema_inventory>

---

## SECTION 2: BEHAVIORAL EXECUTION GUIDELINES & CONSTRAINTS
[CRITICAL] Treat the operational boundaries, semantic definitions, and filter requirements below with the HIGHEST PRIORITY. They dictate how human prompts must manifest as compliant SQL constraints.

### 2.1 SEMANTIC TRANSLATION RULES
<semantic_glossary>
{SEMANTIC_GLOSSARY.strip()}
</semantic_glossary>

### 2.2 COMPLIANCE & BUSINESS POLICIES
<business_context_rules>
{BUSINESS_CONTEXT.strip()}
</business_context_rules>
"""
        
    if not db_awake:
        st.warning("Database failed to wake up within timeout limit. Using static fallback.")
        return build_structured_payload(STATIC_SCHEMA_FALLBACK.strip())

    try:
        # ⚡ Connected to shared connection instance
        with db.engine.connect() as conn:
            # 1. Get all tables in public schema
            tables_result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_type = 'BASE TABLE'
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

            # 4. Assemble the raw catalog component
            schema_catalog_lines = []
            for table in tables:
                cols = table_cols.get(table, [])
                schema_catalog_lines.append(f"### Table: `{table}`")
                schema_catalog_lines.append(f"- **Columns:** {', '.join(cols)}")
                if relationships.get(table):
                    schema_catalog_lines.append(f"- **Foreign Keys:** {', '.join(relationships[table])}")
                schema_catalog_lines.append("")

            schema_catalog_lines.append("### Implicit Dynamic Joins:")
            schema_catalog_lines.append("- Implicit matching pattern: For tables missing formal FK definitions, join using `r.email = target_table.email`.")
            
            raw_catalog = "\n".join(schema_catalog_lines)
            return build_structured_payload(raw_catalog)

    except Exception as e:
        st.warning(f"Could not load live schema, using structured static fallback: {e}")
        return build_structured_payload(STATIC_SCHEMA_FALLBACK.strip())


# ----------------------------
# Schema context
# ----------------------------
BUSINESS_CONTEXT = """COMPLIANCE & BUSINESS POLICIES:
- respondent: Core table containing global panel memberships. Nearly all contextual tables associate through this table.
- Association standard: Most contextual tables establish relations explicitly through the `email` column.
- unsubscribe_blacklist: Contains opted-out panel profiles.

DETERMINISTIC EXECUTION RULES:
1. Exclude opted-out audience pools from all executable SQL outputs and counts by enforcing an anti-join or inclusion check against the `unsubscribe_blacklist` table. 
   Execution Standard: Add `AND r.email NOT IN (SELECT email FROM unsubscribe_blacklist)` or implement a LEFT JOIN where `unsubscribe_blacklist.email IS NULL`.
2. Generate queries using valid PostgreSQL syntax exclusively.
3. Focus columns `is_deleted` and `is_active` within the `respondent` and `respondent_type_specification` tables only when the user's prompt explicitly states to use those columns.
4. In multi-table JOIN operations, explicitly prefix every column name with its declared table alias across all statement fragments, including SELECT, WHERE, ON, GROUP BY, and ORDER BY blocks (e.g., `SELECT r.email`, `SELECT a.country`).
5. Formulate COUNT aggregations with a singular, clearly named column alias (e.g., `SELECT COUNT(DISTINCT r.email) AS total_respondents`).
6. Connect `respondent` to `addresses` utilizing a LEFT JOIN configuration. Reserve INNER JOIN configurations strictly for instances where the prompt introduces mandatory structural address boundaries.
7. Explicitly declare every requested column name individually in the SELECT clause. Avoid using wildcard operators like `*`.
8. Validate date filter comparisons using TIMESTAMP WITH TIME ZONE notation patterns (e.g., `column_name >= '2024-01-01'::timestamptz`). Restrict operations to explicit date columns documented in the physical dictionary.
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
You are an expert AI data assistant specializing in conversation context restoration for a database query system.
Your job is to read the dialogue history and the user's latest question, then rephrase the latest question into a completely standalone, self-contained question.

### REPHRASING RULES:
1. Core Scope Preservation: Retain all operational attributes from recent history (e.g., specific Project Numbers like 'FSI260409IV', countries, gender, or age boundaries) unless the user's latest question explicitly alters those bounds.
2. Pronoun Resolution: Replace ambiguous terms ("they", "them", "those", "the list", "the cohort") with the explicit target entity group established in the conversation history (e.g., "Consumers", "Healthcare Professionals").
3. Intent Continuity: Maintain the core request type—if the user asks "how many", the standalone question must demand a count. If they ask for "a list" or "export", it must demand a detailed breakdown.

### OUTPUT EXPECTATIONS:
You must structure your reply using XML tags to isolate your reasoning from your final answer:
- Write your brief step-by-step history analysis inside <thinking>...</thinking> tags.
- Write the final, completely standalone rephrased question inside <rewritten_question>...</rewritten_question> tags.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

SIMPLE_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """
{schema}

You are an expert PostgreSQL database engineer for a market research firm. 
Provide the highly optimized PostgreSQL query to answer the user's question.
Return ONLY the final SQL query enclosed strictly within ```sql and ``` markdown tags. 
Do not explain the SQL or provide any step-by-step reasoning.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Current Question: {question}")
])

COMPLEX_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """
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
    ("human", "Current Question: {question}")
])

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a data analyst reporting internal database results to a colleague.

Rules:
- Report ONLY what the data shows. Stick strictly to internal data and avoid external benchmarks or real-world statistics.
- For single-value results, respond in one short sentence stating just the number.
- Provide a brief text summary of the findings rather than generating a Markdown table or returning the raw rows.
- Omit commentary, caveats, or explanations unless the data is empty.
- If no data is returned, output exactly: "No results were found for that question."
- Each email address must appear only once in the output. Where an email has multiple associated values in the respondent_type, projects, respondent_hcp_level_of_expertises, conditions, or hcp_job_title columns, consolidate those values into a single cell for that email, using a semicolon (;) as the delimiter.
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
You must evaluate the generated SQL query against the user's original question using the operational guidelines, semantic mappings, and database rules provided below.

### RULES & SEMANTIC CONTEXT:
{rules}

### YOUR EVALUATION TASKS:
Perform these two checks:
1. Filters & Mappings: Does the SQL accurately apply EVERY filter and condition requested in the user's question? Verify that it correctly implements the rules in the SEMANTIC TRANSLATION GLOSSARY and BUSINESS CONTEXT above.
2. Output Format: Does the SQL use the correct aggregation? If the user asks "how many", the SQL MUST use COUNT(). If they ask for a list, selection, or export, it MUST explicitly select email, first_name, last_name, AND EVERY SINGLE COLUMN used in the WHERE clause.

### OUTPUT FORMAT (STRICT):
Your response must strictly follow one of these two formats. Do not include introductory text or conversational fluff.

FORMAT 1: If the SQL is 100% correct and compliant, output exactly this single word:
VALID

FORMAT 2: If the SQL fails any check, output a short, bulleted list of the specific errors found, then STOP generating immediately. 
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

EXPORT_POLICIES = """
### EXPORT & LISTING OVERRIDES
[CRITICAL] The user has explicitly requested a data export, list, or detailed audience extraction. You MUST adhere to these overrides:
1. Output Format: You MUST explicitly select `r.email`, `r.first_name`, `r.last_name`. IN ADDITION, you MUST explicitly include EVERY column you use in your WHERE clause inside your SELECT clause. NEVER use COUNT() for this query.
2. Audience Capture Strategy (The "For Project" Rule): When a user asks to export an audience "for" a project (e.g., "export consumers for project FSI123"), they are building a NEW target list. You MUST NOT join the `projects` or `project_respondent` tables, and you MUST NOT filter by `project_number` in your SQL. ONLY join project tables if the user explicitly asks for people who "already participated", "completed", or "applied" to a specific project in the past.
"""

# ----------------------------
# Core functions
# ----------------------------

def _parse_history(history) -> list:
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
    match = re.search(r"```sql\n(.*?)\n```", llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match_fallback = re.search(r"```( * ? )```", llm_response, re.DOTALL)
    if match_fallback:
        return match_fallback.group(1).strip()

    return llm_response.strip()


def validate_sql_intent(question: str, sql: str, is_export: bool = False) -> str:
    llm = get_llm()
    chain = VALIDATION_PROMPT | llm
    rules_context = f"{SEMANTIC_GLOSSARY}\n\n{BUSINESS_CONTEXT}"

    if is_export:
        rules_context += f"\n\n{EXPORT_POLICIES}"
    
    result = chain.invoke({"rules": rules_context, "question": question, "sql": sql}).content.strip()
    return result


def is_complex_query(question: str) -> bool:
    q_lower = question.lower()
    complexity_indicators = [
        "between", "average", "ratio", "compare", "trend", "month", "year",
        "both", "multiple", "except", "without", "versus", "vs", "most",
        "least", "top", "bottom", "percentage", "group by"
    ]
    if len(q_lower.split()) > 15:
        return True
    if any(word in q_lower for word in complexity_indicators):
        return True
    return False


def run_query(sql: str, max_retries: int = 5, delay: int = 3) -> pd.DataFrame | None:
    """Runs the query using the persistent pre-ping connection pool."""
    db = get_db()
    df = None

    with st.status("Connecting to database...", expanded=True) as status:
        for attempt in range(max_retries):
            try:
                status.update(label="Executing query...", state="running")
                # ⚡ Grab a hot socket from the shared connection manager
                with db.engine.connect() as conn:
                    result = conn.execute(text(sql))
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    status.update(label="Query successful!", state="complete", expanded=False)
                    break 

            except Exception as e:
                error_str = str(e).lower()
                is_conn_error = any(keyword in error_str for keyword in [
                    "connection", "timeout", "closed", "ssl", "operationalerror"
                ])

                if is_conn_error and attempt < max_retries - 1:
                    status.update(
                        label=f"Re-verifying bridge pool... (Attempt {attempt + 1} of {max_retries})",
                        state="running"
                    )
                    time.sleep(delay)
                else:
                    status.update(label="Query failed.", state="error")
                    st.session_state["last_sql_error"] = str(e)
                    st.error(f"SQL execution error: {e}")
                    break

    return df


def get_column_samples(sql: str) -> str:
    """Fetches distinct string values for diagnostic retries using the shared manager."""
    db = get_db()
    samples = []

    words = sql.lower().split()
    tables = []
    for i, word in enumerate(words):
        if word in ("from", "join") and i + 1 < len(words):
            table = words[i + 1].strip("(),;")
            if table and not table.startswith("("):
                tables.append(table)

    try:
        with db.engine.connect() as conn:
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
    """Extracts table names from failed SQL and streams columns back from info_schema."""
    db = get_db()
    lines = []

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
        with db.engine.connect() as conn:
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

def detect_export_request(question: str) -> tuple[bool, str | None, bool]:
    has_export = bool(EXPORT_KEYWORD_PATTERN.search(question))
    project_match = PROJECT_NUMBER_PATTERN.search(question)
    project_number = project_match.group(0).lower() if project_match else None
    is_export = has_export and project_number is not None

    is_override = False
    if is_export:
        q_lower = question.lower()
        is_override = any(phrase in q_lower for phrase in EXPORT_OVERRIDE_PHRASES)

    return is_export, project_number, is_override

def extract_and_strip_limit(sql: str) -> tuple[str, int | None]:
    """
    Detects a trailing LIMIT n clause, strips it, and returns the cleaned SQL
    together with the limit value (None if no LIMIT was present). Used to move
    LIMIT from the inner query to the outer export-exclusion wrapper so the cap
    applies AFTER previously-exported emails are excluded, not before.
    """
    cleaned = sql.strip().rstrip(';').strip()
    match = _TRAILING_LIMIT_RE.search(cleaned)
    if match:
        limit_value = int(match.group(1))
        cleaned = _TRAILING_LIMIT_RE.sub('', cleaned).strip()
        return cleaned, limit_value
    return cleaned, None

def add_export_exclusion_to_sql(sql: str, project_number: str) -> str:
    cleaned_sql, limit_value = extract_and_strip_limit(sql)

    wrapped_sql = (
        f"SELECT * FROM (\n"
        f"    {cleaned_sql}\n"
        f") AS final_output\n"
        f"WHERE final_output.email NOT IN (\n"
        f"    SELECT email FROM export_tracker\n"
        f"    WHERE project_number = '{project_number.lower()}'\n"
        f")"
    )

    if limit_value is not None:
        wrapped_sql += f"\nLIMIT {limit_value}"

    return wrapped_sql


def extract_base_sql_for_storage(sql: str) -> str:
    match = re.search(
        r"SELECT \* FROM \(\s*(.*?)\s*\)\s*AS final_output\s*WHERE final_output\.email NOT IN", 
        sql, 
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return sql.strip()

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
    base_sql = extract_base_sql_for_storage(sql)

    # Drop ORDER BY and everything trailing it
    base_no_order = re.sub(
        r'\bORDER\s+BY\b.*$', '', base_sql, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    base_no_order = re.sub(                                           
    r'\bLIMIT\s+\d+\s*;?\s*$', '', base_no_order, flags=re.IGNORECASE
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

def get_already_exported_count(project_number: str) -> int:
    """Total emails already tracked in export_tracker for a given project."""
    db = get_db()
    try:
        # ⚡ Switched to clean single-line scalar checkout
        result = db.execute("SELECT COUNT(*) FROM export_tracker WHERE project_number = :pn", {"pn": project_number})
        return result.scalar() or 0
    except Exception as e:
        st.warning(f"Could not query export_tracker: {e}")
        return 0


def insert_export_tracking(
    emails: list[str],
    project_number: str,
    filters_str: str,
    fingerprint: str,
    is_override: bool = False,
) -> int:
    """Bulk-inserts emails using a self-committing block transaction context."""
    if not emails:
        return 0

    conflict = (
        "DO UPDATE SET filters = EXCLUDED.filters, "
        "filter_fingerprint = EXCLUDED.filter_fingerprint, "
        "datetimestamp = EXCLUDED.datetimestamp"
        if is_override else "DO NOTHING"
    )

    db = get_db()

    try:
        with db.engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO export_tracker
                        (email, project_number, filters, filter_fingerprint, datetimestamp)
                    SELECT
                        unnest(CAST(:emails AS varchar[])),
                        :pn,
                        :filters,
                        :fingerprint,
                        NOW() AT TIME ZONE 'UTC'
                    ON CONFLICT (email, project_number) {conflict}
                """),
                {
                    "emails": emails,
                    "pn": project_number,
                    "filters": filters_str,
                    "fingerprint": fingerprint,
                },
            )
    except Exception as e:
        st.error(f"Export tracking insert failed: {e}")
        return 0

    return len(emails)


def _handle_export_tracking(sql: str, df, project_number: str | None, is_export: bool, is_override: bool) -> None:
    if not is_export or not project_number or df is None or df.empty:
        return

    email_col = next((c for c in df.columns if 'email' in c.lower()), None)
    if email_col is None:
        st.warning("⚠️ Export detected but result has no email column — tracking skipped.")
        return

    emails = df[email_col].dropna().unique().tolist()
    filters_str = extract_base_sql_for_storage(sql)
    fingerprint = compute_filter_fingerprint(sql) 

    with st.spinner(f"Tracking {len(emails)} records for project {project_number.upper()}..."):
        written = insert_export_tracking(emails, project_number, filters_str, fingerprint, is_override)

    action = "exported / refreshed" if is_override else "newly exported"
    skipped = len(emails) - written if not is_override else 0

    msg = f"📤 **Export Tracker** — `{project_number}`: **{written}** email(s) {action}."
    if skipped > 0:
        msg += f" _{skipped} skipped (already tracked)._"
    st.success(msg)


def contextualize_user_question(question: str, history: list) -> str:
    if not history:
        return question

    parsed_history = _parse_history(history)
    llm = get_llm()
    chain = CONTEXTUALIZE_PROMPT | llm
    
    try:
        response = chain.invoke({
            "history": parsed_history,
            "question": question
        }).content.strip()
        
        match = re.search(r"<rewritten_question>(.*?)</rewritten_question>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        clean_text = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL).strip()
        if clean_text:
            return clean_text
            
    except Exception as e:
        st.warning(f"Contextualization pipeline anomaly: {e}. Defaulting to raw input.")
        
    return question


def generate_sql_with_retry(question: str, history: list = None) -> tuple[str | None, pd.DataFrame | None, list]:

    query_process_logs = []
    
    # Elegant layout logging interceptor
    def log_ui(func_name: str, *args, **kwargs):
        if hasattr(st, func_name):
            getattr(st, func_name)(*args, **kwargs)
        query_process_logs.append({"type": func_name, "args": args, "kwargs": kwargs})
        
    raw_original_question = question
    question = contextualize_user_question(question, history)
    
    parsed_history = _parse_history(history)
    is_export, project_number, is_override = False, None, False

    with st.expander("🔍 Query Process", expanded=True):
        log_ui("markdown", "**Step 0: Synchronizing Context ...**")
        if parsed_history:
            if question != raw_original_question:
                log_ui("info", f"🔄 Rephrased contextually to: *\"{question}\"*")
            else:
                log_ui("info", "📊 Context analyzed; query contains explicit attributes.")
        else:
            log_ui("info", "🆕 Fresh chat thread started.")
            
        is_export, project_number, is_override = detect_export_request(question)

        log_ui("markdown", "**Step 1: Generating SQL...**")
        sql = generate_sql(question, history=history, is_export=is_export)

        if not sql:
            log_ui("error", "Could not generate a valid SQL query.")
            return None, None, query_process_logs

        valid_sql_starts = ("select", "with", "(", "--", "/*")
        if not sql.lower().startswith(valid_sql_starts):
            log_ui("info", "No database query required for this message.")
            return sql, None, query_process_logs

        log_ui("code", sql, language="sql")

        log_ui("markdown", "**Step 2: Validating Query Intent...**")
        validation_result = validate_sql_intent(question, sql, is_export=is_export)

        if validation_result != "VALID":
            log_ui("warning", f"⚠️ Code Reviewer flagged an issue: {validation_result}")
            log_ui("markdown", "**Step 2.5: Rewriting SQL to include missing parameters...**")

            rewrite_rules = f"{SEMANTIC_GLOSSARY}\n\n{BUSINESS_CONTEXT}"
            if is_export:
                rewrite_rules += f"\n\n{EXPORT_POLICIES}"

            llm = get_llm()
            chain = REWRITE_PROMPT | llm
            sql = chain.invoke({
                "rules": rewrite_rules,
                "history": parsed_history,
                "question": question,
                "bad_sql": sql,
                "missing_logic": validation_result
            }).content.strip()

            log_ui("code", sql, language="sql")
        else:
            log_ui("info", "✅ Query passed intent validation.")

        if is_export and not is_override:
            already_count = get_already_exported_count(project_number)
            log_ui("markdown", "**Export Tracking: Excluding previously exported emails...**")
            if already_count > 0:
                log_ui("info",
                    f"ℹ️ {already_count} previously exported email(s) for "
                    f"`{project_number}` will be excluded from this run."
                )
            sql = add_export_exclusion_to_sql(sql, project_number)
            log_ui("markdown", "**Modified SQL (export exclusion applied):**")
            log_ui("code", sql, language="sql")
        elif is_export and is_override:
            log_ui("info", f"🔄 Export override active — all matching emails will be included for `{project_number}`.")

        log_ui("markdown", "**Step 3: Running query...**")
        df = run_query(sql)
        sql_error = st.session_state.pop("last_sql_error", None)

        if df is not None and not df.empty:
            log_ui("success", f"Query returned {len(df)} row(s). No retry needed.")
            _handle_export_tracking(sql, df, project_number, is_export, is_override)
            return sql, df, query_process_logs

        if (df is not None and df.empty) or (df is None and sql_error):
            if sql_error and "undefinedcolumn" in sql_error.lower():
                log_ui("warning", "⚠️ Query used a column that doesn't exist. Fetching real column names...")
                log_ui("markdown", "**Step 4: Injecting real column inventory...**")

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

                    log_ui("markdown", "**Step 4: Retried SQL:**")
                    log_ui("code", sql, language="sql")

                    df = run_query(sql)
                    sql_error = st.session_state.pop("last_sql_error", None)
                    if df is not None and not df.empty:
                        log_ui("success", f"✅ Retry successful — {len(df)} row(s).")
                    elif sql_error:
                        log_ui("error", f"Retry also failed: {sql_error}")
                    else:
                        log_ui("info", "Retry ran but returned 0 results.")
                    _handle_export_tracking(sql, df, project_number, is_export, is_override)
                    return sql, df, query_process_logs

            else:
                if sql_error:
                    log_ui("warning", f"⚠️ Query failed: {sql_error}")
                else:
                    log_ui("warning", "⚠️ Query returned 0 results. Checking actual stored values...")

                log_ui("markdown", "**Step 3: Fetching column value samples...**")
                samples = get_column_samples(sql)

                if samples:
                    log_ui("info", "✅ Retrieved column value samples for retry.")
                    log_ui("markdown", "**Step 4: Retrying with correct values...**")

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

                    log_ui("markdown", "**Step 4: Retried SQL:**")
                    log_ui("code", sql, language="sql")

                    df = run_query(sql)
                    sql_error = st.session_state.pop("last_sql_error", None)
                    if df is not None and not df.empty:
                        log_ui("success", f"✅ Retry successful — {len(df)} row(s).")
                    elif sql_error:
                        log_ui("error", f"Retry also failed: {sql_error}")
                    else:
                        log_ui("info", "Retry ran but returned 0 results.")
                else:
                    log_ui("warning", "Could not fetch column samples for retry.")

    _handle_export_tracking(sql, df, project_number, is_export, is_override)
    return sql, df, query_process_logs


def generate_sql(question: str, history: list = None, is_export: bool = False) -> str | None:
    if history is None:
        history = []

    llm = get_llm()
    requires_cot = is_complex_query(question)
    prompt_template = COMPLEX_SQL_PROMPT if requires_cot else SIMPLE_SQL_PROMPT
    
    system_prompt = get_schema_description()
    if is_export:
        system_prompt += f"\n\n{EXPORT_POLICIES}"
        
    chain = prompt_template | llm

    try:
        raw_response = chain.invoke({
            "schema": system_prompt,
            "history": history,
            "question": question
        }).content

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
    if not sql:
        return False
    valid_sql_starts = ("select", "with", "(", "--", "/*")
    return sql.strip().lower().startswith(valid_sql_starts)


@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    return run_query(sql)


def should_generate_chart_cached(question, sql, df):
    return False 


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
        cols = ", ".join(df.columns.tolist())
        return f"Query returned {len(df)} records with columns: {cols}."
