import time
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from fsi_ai import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached
)
from operations import show_operations_page
from database import get_db

db = get_db()
try:
    db.ensure_awake()
except Exception as e:
    st.error("⚠️ Failed to establish persistent database bridge on startup.")

def build_history_messages(messages: list, max_turns: int = 3) -> list:
    """
    Builds conversation context from the last N turns.
    Skips history if the current question appears to be a topic shift
    (i.e. doesn't reference pronouns or connective words).
    """
    if not messages:
        return []

    # Connective words that signal the user is continuing a prior thread
    continuation_signals = [
        "and ", "also ", "what about", "how about", "same for", 
        "now ", "but ", "instead", "those", "them", "their",
        "that", "these", "the same", "similar", "above"
    ]
    
    # Get the last user message to check if it's a continuation
    last_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"].lower()
            break
    
    is_continuation = any(signal in last_user_msg for signal in continuation_signals)
    
    # If no continuation signal, this looks like a fresh topic — skip history
    if not is_continuation:
        return []
        
    pairs = []
    i = len(messages) - 1
    while i >= 0 and len(pairs) < max_turns:
        if messages[i]["role"] == "assistant" and i > 0 and messages[i-1]["role"] == "user":
            q = messages[i-1]["content"]
            sql = messages[i].get("sql", "")
            pairs.append((q, sql))
            i -= 2
        else:
            i -= 1
    
    if not pairs:
        return []
    
    # Build structured LangChain message objects
    langchain_messages = []
    for q, sql in reversed(pairs):
        # 1. Add the human's past question
        langchain_messages.append(HumanMessage(content=q))
        
        # 2. Add the AI's past SQL response
        if sql:
            # We wrap the SQL in markdown tags so it identically matches 
            # the format the Llama 3 model is instructed to output.
            langchain_messages.append(AIMessage(content=f"```sql\n{sql}\n```"))
            
    return langchain_messages

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

avatar_url = "https://i0.wp.com/fieldscopeint.com/wp-content/uploads/2026/03/logo-FSI.jpg?resize=200%2C200&ssl=1"

st.set_page_config(layout="wide")

# ------------------------------------------------------------
# PERMANENT NARROW SIDEBAR NAVIGATION
# ------------------------------------------------------------

# Inject custom CSS to narrow down the sidebar width permanently
st.markdown(
    """
    <style>
        /* Forces the sidebar navigation container width to stay narrow */
        [data-testid="stSidebar"] {
            min-width: 240px !important;
            max-width: 240px !important;
        }
        
        /* Overrides the overly bright green selection color for primary buttons */
        button[data-testid="stBaseButton-primary"] {
            background-color: #1E3A8A !important;  /* Elegant dark blue/navy */
            border-color: #1E3A8A !important;
            color: #ffffff !important;
        }
        
        /* Optional: Change hover styling for primary buttons */
        button[data-testid="stBaseButton-primary"]:hover {
            background-color: #172554 !important;
            border-color: #172554 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize navigation page state
if "page" not in st.session_state:
    st.session_state["page"] = "FSI AI"

st.sidebar.title("Navigation")

# Navigation buttons
if st.sidebar.button("🤖 FSI AI", use_container_width=True, type="primary" if st.session_state["page"] == "FSI AI" else "secondary"):
    st.session_state["page"] = "FSI AI"
    st.rerun()

if st.sidebar.button("📊 Operations", use_container_width=True, type="primary" if st.session_state["page"] == "Operations" else "secondary"):
    st.session_state["page"] = "Operations"
    st.rerun()


# ------------------------------------------------------------
# PAGE ROUTING INTERFACES
# ------------------------------------------------------------

if st.session_state["page"] == "FSI AI":
    st.title("FSI AI")

    def set_question(question):
        st.session_state["my_question"] = question

    # 1. Initialize the chat history list if it doesn't exist yet
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 2. Display the "Suggested Questions" button ONLY if the chat is empty
    if len(st.session_state["messages"]) == 0:
        assistant_message_suggested = st.chat_message("assistant", avatar=avatar_url)
        if assistant_message_suggested.button("Click to show suggested questions"):
            questions = generate_questions_cached()
            for i, question in enumerate(questions):
                time.sleep(0.05)
                st.button(question, on_click=set_question, args=(question,))

    # 3. Loop through and draw all past messages
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar=avatar_url):
                if msg.get("error"):
                    st.error(msg["error"])
                else:
                    # Defaulting global flags to True/False explicitly now that toggles are removed
                    if msg.get("sql"):
                        st.code(msg["sql"], language="sql", line_numbers=True)
                        
                    if msg.get("df") is not None:
                        df = msg["df"]
                        # Provide the download button for the FULL dataframe
                        csv = convert_df_to_csv(df)
                        st.download_button(
                             label=f"📥 Download Full Data ({len(df)} rows)",
                             data=csv,
                             file_name='fsi_data_export.csv',
                             mime='text/csv',
                             key=f"download_hist_{uuid.uuid4()}" 
                        )
                        if len(df) > 10:
                            st.caption(f"Showing first 10 of {len(df)} rows below:")
                            st.dataframe(df.head(10))
                        else:
                            st.dataframe(df)
                            
                    if msg.get("plotly_code"):
                        st.code(msg["plotly_code"], language="python", line_numbers=True)
                    if msg.get("fig"):
                        st.plotly_chart(msg["fig"])
                    if msg.get("summary"):
                        st.text(msg["summary"])

    # 4. Always show the input box
    user_input = st.chat_input("Ask me a question about your data")

    # Determine the current question (from chat input OR from clicking a suggestion)
    my_question = None
    if user_input:
        my_question = user_input
    elif st.session_state.get("my_question"):
        my_question = st.session_state["my_question"]
        # Clear it so it doesn't trigger again on the next UI rerun
        st.session_state["my_question"] = None 

    # 5. Process the NEW question
    if my_question:
        # Append user question to history
        st.session_state["messages"].append({"role": "user", "content": my_question})
        st.chat_message("user").write(my_question)
        
        # Process assistant response inside its chat bubble
        with st.chat_message("assistant", avatar=avatar_url):
            # We'll build a dictionary to save this turn's data to history
            turn_data = {"role": "assistant"}

            history_msgs = build_history_messages(st.session_state["messages"][:-1])
            sql, df = generate_sql_cached(question=my_question, history=history_msgs)
            
            if sql and is_sql_valid_cached(sql=sql):
                turn_data["sql"] = sql
                st.code(sql, language="sql", line_numbers=True)
                
                if df is not None:
                    turn_data["df"] = df

                    # Custom Download Button for the Active Turn
                    csv = convert_df_to_csv(df)
                    st.download_button(
                         label=f"📥 Download Full Data ({len(df)} rows)",
                         data=csv,
                         file_name='fsi_data_export.csv',
                         mime='text/csv',
                         key=f"download_active_{uuid.uuid4()}" 
                    )
                    
                    if len(df) > 10:
                        st.caption(f"Showing first 10 of {len(df)} rows below:")
                        st.dataframe(df.head(10))
                    else:
                        st.dataframe(df)
                            
                    if should_generate_chart_cached(question=my_question, sql=sql, df=df):
                        code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)
                        turn_data["plotly_code"] = code
                        if code:
                            fig = generate_plot_cached(code=code, df=df)
                            if fig:
                                turn_data["fig"] = fig
                                st.plotly_chart(fig)
                            else:
                                st.error("I couldn't generate a chart")
                                
                    summary = generate_summary_cached(question=my_question, df=df)
                    if summary:
                        turn_data["summary"] = summary
                        st.text(summary)
            else:
                turn_data["error"] = "I wasn't able to generate SQL for that question or the query was unsupported."
                st.error(turn_data["error"])
                
            st.session_state["messages"].append(turn_data)

elif st.session_state["page"] == "Operations":
    # Call the functional module code directly to draw the dashboard layout
    show_operations_page()
