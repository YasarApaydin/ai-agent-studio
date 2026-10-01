import streamlit as st
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


@st.cache_resource
def create_agent_with_ollama():
    return create_agent(model=ChatOllama(model="qwen3:1.7b", temperature=0), tools=[])


agent = create_agent_with_ollama()

st.set_page_config(page_title="LangChain Ollama Agent", page_icon=":robot_face:")
st.title("Simple Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for mesaj in st.session_state.messages:
    with st.chat_message(mesaj["role"]):
        st.markdown(mesaj["content"])



if prompt := st.chat_input("Ask me a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.invoke({"messages": st.session_state.messages})
            output = response["messages"][-1].content
        st.markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})
     