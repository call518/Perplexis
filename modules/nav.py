import streamlit as st

def Navbar():
    with st.sidebar:
        st.page_link('main.py', label='Home', icon=':material/home:')
        st.page_link('pages/chat_and_rag.py', label='RAG & Chat', icon=':material/quick_reference_all:')