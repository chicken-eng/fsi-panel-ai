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

# 💡 Initialize conversation states securely at startup
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "staged_edit_prompt" not in st.session_state:
    st.session_state["staged_edit_prompt"] = None 

if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

db = get_db()
try:
    db.ensure_awake()
except Exception as e:
    st.error("⚠️ Failed to establish persistent database bridge on startup.")

def build_history_messages(messages: list, max_turns: int = 3) -> list:
    """
    Builds conversation context from the last N turns.
    Skips history if the current question appears to be a topic shift.
    """
    if not messages:
        return []

    continuation_signals = [
        "and ", "also ", "what about", "how about", "same for", 
        "now ", "but ", "instead", "those", "them", "their",
        "that", "these", "the same", "similar", "above"
    ]
    
    last_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"].lower()
            break
    
    is_continuation = any(signal in last_user_msg for signal in continuation_signals)
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
    
    langchain_messages = []
    for q, sql in reversed(pairs):
        langchain_messages.append(HumanMessage(content=q))
        if sql:
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
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 240px !important;
            max-width: 240px !important;
        }
        button[data-testid="stBaseButton-primary"] {
            background-color: #1E3A8A !important;  
            border-color: #1E3A8A !important;
            color: #ffffff !important;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background-color: #172554 !important;
            border-color: #172554 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if "page" not in st.session_state:
    st.session_state["page"] = "FSI AI"

st.sidebar.title("Navigation")

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

    # Display the "Suggested Questions" button ONLY if the chat is empty
    if len(st.session_state["messages"]) == 0:
        assistant_message_suggested = st.chat_message("assistant", avatar=avatar_url)
        if assistant_message_suggested.button("Click to show suggested questions"):
            questions = generate_questions_cached()
            for i, question in enumerate(questions):
                time.sleep(0.05)
                st.button(question, on_click=set_question, args=(question,))

    # Loop through and draw all past messages
    for idx, msg in enumerate(st.session_state["messages"]):
        if msg["role"] == "user":
            with st.chat_message("user"):
                col_text, col_edit = st.columns([0.96, 0.04])
                col_text.write(msg["content"])
                
                # Simple clean edit trigger icon button
                if col_edit.button("✏️", key=f"edit_btn_{idx}", help="Edit this question"):
                    st.session_state["edit_index"] = idx
                    st.rerun()
            
            # 💡 CHANGE 1: THE BIGGER, CENTERED INLINE EDIT WINDOW
            # Appears beautifully framed directly underneath the targeted user question block
            if st.session_state["edit_index"] == idx:
                edit_layout_col1, edit_layout_col2, edit_layout_col3 = st.columns([0.05, 0.90, 0.05])
                with edit_layout_col2:
                    with st.container(border=True):
                        st.markdown("### ✏️ Edit Your Question")
                        edited_prompt = st.text_area(
                            "Modify prompt text below:", 
                            value=msg["content"], 
                            height=160,  # Generous height expansion
                            key=f"edit_input_{idx}"
                        )
                        act_col1, act_col2 = st.columns([0.15, 0.85])
                        if act_col1.button("Cancel", key=f"cancel_edit_{idx}", use_container_width=True):
                            st.session_state["edit_index"] = None
                            st.rerun()
                        if act_col2.button("Apply Changes & Rerun From Here", key=f"run_edit_{idx}", type="primary", use_container_width=True):
                            st.session_state["messages"] = st.session_state["messages"][:idx]
                            st.session_state["staged_edit_prompt"] = edited_prompt
                            st.session_state["edit_index"] = None
                            st.rerun()
                            
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar=avatar_url):
                if msg.get("error"):
                    st.error(msg["error"])
                else:
                    if msg.get("sql"):
                        st.code(msg["sql"], language="sql", line_numbers=True)
                        
                    if msg.get("df") is not None:
                        df = msg["df"]
                        csv = convert_df_to_csv(df)
                        st.download_button(
                             label=f"📥 Download Full Data ({len(df)} rows)",
                             data=csv,
                             file_name='fsi_data_export.csv',
                             mime='text/csv',
                             key=f"download_hist_{idx}" 
                        )
                        if len(df) > 10:
                            st.caption(f"Showing first 10 of {len(df)} rows below:")
                            st.dataframe(df.head(10))
                        else:
                            st.dataframe(df)
                            
                    # 💡 CHANGE 2: FIXED INDENTATION 
                    # These steps now render accurately and independently outside of the DataFrame block
                    if msg.get("plotly_code"):
                        st.code(msg["plotly_code"], language="python", line_numbers=True)
                    if msg.get("fig"):
                        st.plotly_chart(msg["fig"])
                    if msg.get("summary"):
                        st.text(msg["summary"])

    # Always show the input box at the bottom
    user_input = st.chat_input("Ask me a question about your data")

    my_question = None
    if st.session_state.get("staged_edit_prompt"):
        my_question = st.session_state["staged_edit_prompt"]
        st.session_state["staged_edit_prompt"] = None  
    elif user_input:
        my_question = user_input
    elif st.session_state.get("my_question"):
        my_question = st.session_state["my_question"]
        st.session_state["my_question"] = None 

    # Process the active question payload
    if my_question:
        st.session_state["messages"].append({"role": "user", "content": my_question})
        st.chat_message("user").write(my_question)
        
        with st.chat_message("assistant", avatar=avatar_url):
            turn_data = {"role": "assistant"}

            history_msgs = build_history_messages(st.session_state["messages"][:-1])
            sql, df = generate_sql_cached(question=my_question, history=history_msgs)
            
            if sql and is_sql_valid_cached(sql=sql):
                turn_data["sql"] = sql
                st.code(sql, language="sql", line_numbers=True)
                
                if df is not None:
                    turn_data["df"] = df
                    csv = convert_df_to_csv(df)
                    st.download_button(
                         label=f"📥 Download Full Data ({len(df)} rows)",
                         data=csv,
                         file_name='fsi_data_export.csv',
                         mime='text/csv',
                         key=f"download_active_{len(st.session_state['messages'])}" 
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
            st.rerun()  

elif st.session_state["page"] == "Operations":
    show_operations_page()
