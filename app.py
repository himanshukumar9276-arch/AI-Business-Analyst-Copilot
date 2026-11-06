import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from graph.graph_builder import graph


st.set_page_config(
    page_title="AI Business Analyst Copilot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Business Copilot")
st.caption("SQL • RAG • Web Search • Conversational Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


with st.sidebar:

    st.header("Conversation")

    st.text_input(
        "Thread ID",
        value=st.session_state.thread_id,
        disabled=True
    )

    if st.button("🆕 New Chat", use_container_width=True):

        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()

    st.divider()

    st.subheader("Capabilities")

    st.success("SQL Agent")
    st.success("RAG Agent")
    st.success("Web Agent")
    st.success("Conversation Memory")



for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])



prompt = st.chat_input("Ask me anything...")


if prompt:


    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    state = {
        "messages": [
            HumanMessage(content=prompt)
        ]
    }

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = graph.invoke(
                    state,
                    config=config
                )

                response = result["messages"][-1].content

            except Exception as e:

                response = f"❌ Error:\n\n{e}"

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )