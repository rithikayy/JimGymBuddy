import streamlit as st
from app import create_rag_agent

st.title("Chat with Jim, your Gym Buddy")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'agent' not in st.session_state:
    st.session_state.agent = create_rag_agent()

for msg in st.session_state.messages:
    st.chat_message(msg['role']).markdown(msg['content'])

prompt = st.chat_input('How is it going?')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})

    with st.chat_message('assistant'):
        response = ""
        agent_messages = []
        for msg in st.session_state.messages:
            agent_messages.append({'role':msg['role'], 'content':msg['content']})
        
        for step in st.session_state.agent.stream(
            {"messages": agent_messages},
            stream_mode="values",
        ):
            response = step['messages'][-1].content
        
        st.markdown(response)
        st.session_state.messages.append({'role':'assistant', 'content':response})


