import time
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from vanna_calls import (
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

st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
st.sidebar.checkbox("Show Summary", value=True, key="show_summary")
st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")
st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True)

st.title("FSI AI")
# st.sidebar.write(st.session_state)

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
                if msg.get("sql") and st.session_state.get("show_sql", True):
                    st.code(msg["sql"], language="sql", line_numbers=True)
                    
                if msg.get("df") is not None and st.session_state.get("show_table", True):
                    df = msg["df"]
                    # 1. Provide the download button for the FULL dataframe
                    csv = convert_df_to_csv(df)
                    st.download_button(
                         label=f"📥 Download Full Data ({len(df)} rows)",
                         data=csv,
                         file_name='fsi_data_export.csv',
                         mime='text/csv',
                         key=f"download_hist_{uuid.uuid4()}" # Unique key required for Streamlit buttons
                    )
                    if len(df) > 10:
                        st.caption(f"Showing first 10 of {len(df)} rows below:")
                        st.dataframe(df.head(10))
                    else:
                        st.dataframe(df)
                        
                if msg.get("plotly_code") and st.session_state.get("show_plotly_code", False):
                    st.code(msg["plotly_code"], language="python", line_numbers=True)
                if msg.get("fig") and st.session_state.get("show_chart", True):
                    st.plotly_chart(msg["fig"])
                if msg.get("summary") and st.session_state.get("show_summary", True):
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
            if st.session_state.get("show_sql", True):
                st.code(sql, language="sql", line_numbers=True)
            
            if df is not None:
                turn_data["df"] = df
                if st.session_state.get("show_table", True):

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
                    if st.session_state.get("show_plotly_code", False):
                        st.code(code, language="python", line_numbers=True)
                        
                    if code:
                        fig = generate_plot_cached(code=code, df=df)
                        if fig:
                            turn_data["fig"] = fig
                            if st.session_state.get("show_chart", True):
                                st.plotly_chart(fig)
                        else:
                            st.error("I couldn't generate a chart")
                            
                summary = generate_summary_cached(question=my_question, df=df)
                if summary:
                    turn_data["summary"] = summary
                    if st.session_state.get("show_summary", True):
                        st.text(summary)
                        
                # Follow-up questions (we don't save these to history, just show them for the active turn)
                if st.session_state.get("show_followup", True):
                    followup_questions = generate_followup_cached(question=my_question, sql=sql, df=df)
                    if followup_questions:
                        st.text("Here are some possible follow-up questions")
                        for q in followup_questions[:5]:
                            st.button(q, on_click=set_question, args=(q,))
        else:
            turn_data["error"] = "I wasn't able to generate SQL for that question or the query was unsupported."
            st.error(turn_data["error"])
            
        # Finally, append the full assistant response to the history
        st.session_state["messages"].append(turn_data)
