from dotenv import load_dotenv
import os
from langchain.tools import tool
from tavily import TavilyClient
from typing import Dict,Any
from langchain.agents import create_agent
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage,AIMessage
from langchain_groq import ChatGroq
load_dotenv()  


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

tavily_client = TavilyClient()

@tool
def web_search(query: str)->Dict[str,Any]:
    """Search the web for information related to the query"""
    return tavily_client.search(query)



@st.cache_resource
def get_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=("Sen yardımsever bir asistansın."
        "Güncel bilgi gerektiren sorular için web_search tool'unu kullan."),
        checkpointer=InMemorySaver(),
    )


agent = get_agent()

def stream_response(prompt: str,config:dict):
    for chunk , _meta in agent.stream(
        {"messages":[HumanMessage(content=prompt)]},config, stream_mode="messages"):
        if isinstance(chunk,AIMessage) and chunk.content:
            yield chunk.content



st.set_page_config(page_title="AI Agent")
st.title("Hafızalı AI Agent")
config= {"configurable":{"thread_id":"streamlit_session"}}

#Geçmişi sıfırdan başlat
if "history" not in st.session_state:
    st.session_state.history = []

#Önceki mesajları yazdır
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

if prompt := st.chat_input("Mesajınızı yazın"):
    #kullanıcı mesajını hafızaya ekle
    st.session_state.history.append({"role":"user","text":prompt})

    #kullanıcı mesajını ekranda göster
    with st.chat_message("user"):
        st.markdown(prompt)

    #agent cevabını geldikçe göster
    with st.chat_message("assistant"):
        answer =  st.write_stream(stream_response(prompt,config))

    #agent cevabını geçmişe ekle
    st.session_state.history.append({"role":"assistant","text": answer})