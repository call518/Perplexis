import streamlit as st

def Navbar():
    with st.sidebar:
        st.page_link('main.py', label='Home', icon=':material/home:')
        st.page_link('pages/chat.py', label='Chat', icon=':material/chat_bubble:')
        st.page_link('pages/rag.py', label='RAG', icon=':material/quick_reference_all:')