import streamlit as st
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

llm = ChatOllama( model="qwen3:1.7b", temperature=0)

agent = create_agent(model=llm,tools=[])

question = st.text_input("Question:")

if question:
    result = agent.invoke({"messages": [{"role": "user","content": question}]})
    st.write(result["messages"][-1].content)    
