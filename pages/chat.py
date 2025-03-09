__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from modules.nav import Navbar
from modules.common_functions import (
    telegramSendMessage,
    get_ai_role_and_sysetm_prompt,
    get_country_name_by_code,
    get_max_value_of_model_max_tokens,
    get_max_value_of_model_num_ctx,
    get_max_value_of_model_num_predict,
    get_max_value_of_model_embedding_dimensions
)

#--------------------------------------------------

import streamlit as st
# from streamlit_chat import message
from streamlit_js_eval import streamlit_js_eval
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain

from langchain_community.chat_message_histories import ChatMessageHistory

import bs4
import asyncio

from langchain_community.document_loaders import WebBaseLoader

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_ollama import OllamaLLM

### (임시) OllamaEmbeddings 모듈 임포트 수정 (langchain_ollama 는 임베딩 실패 발생)
### --> langchain_ollama.embeddings 으로 정상화 테스트 중...
# from langchain_ollama import OllamaEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import re
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_community.vectorstores import Chroma

from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
import chromadb
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader

from googlesearch import search

import os
import shutil
import fitz
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time

from langchain.callbacks.base import BaseCallbackHandler  # 추가

#--------------------------------------------------

# 페이지 정보 정의
st.set_page_config(page_title="Perplexis:Chat", page_icon=":books:", layout="wide")
st.title(":material/chat_bubble: _:red[Perplexis]_ Chat")

### Mandatory Keys 설정
### 1. secretes.toml 파일에 설정된 KEY 값이 최우선 적용.
### 2. secretes.toml 파일에 설정된 KEY 값이 없을 경우, os.environ 환경변수로 설정된 KEY 값이 적용.
### 3. secretes.toml 파일과 os.environ 환경변수 모두 설정되지 않은 경우, Default 값을 적용.
def set_env_vars():
    """
    환경 변수를 설정하는 함수입니다.
    1) st.secrets 에서 우선적으로 키를 검색
    2) 환경 변수(os.environ)이 있으면 사용
    3) 두 곳 모두 없을 시, 기본값으로 설정
    """
    env_map = {
        "OLLAMA_BASE_URL": {"secret_key": "OLLAMA_BASE_URL", "default_value": "http://localhost:11434"},
        "OPENAI_BASE_URL": {"secret_key": "OPENAI_BASE_URL", "default_value": "https://api.openai.com/v1"},
        "OPENAI_API_KEY": {"secret_key": "OPENAI_API_KEY", "default_value": ""},
        "PINECONE_API_KEY": {"secret_key": "PINECONE_API_KEY", "default_value": ""},
        "PGVECTOR_HOST": {"secret_key": "PGVECTOR_HOST", "default_value": "localhost"},
        "PGVECTOR_PORT": {"secret_key": "PGVECTOR_PORT", "default_value": "5432"},
        "PGVECTOR_USER": {"secret_key": "PGVECTOR_USER", "default_value": "perplexis"},
        "PGVECTOR_PASS": {"secret_key": "PGVECTOR_PASS", "default_value": "changeme"},
        "LANGCHAIN_API_KEY": {"secret_key": "LANGCHAIN_API_KEY", "default_value": ""},
    }
    for k, v in env_map.items():
        if st.secrets["KEYS"].get(v["secret_key"]):
            os.environ[k] = st.secrets["KEYS"].get(v["secret_key"])
        elif not os.environ.get(k):
            os.environ[k] = v["default_value"]

set_env_vars()

if os.environ["LANGCHAIN_API_KEY"]:
    os.environ["LANGCHAIN_TRACING_V2"]= "true"
    os.environ["LANGCHAIN_ENDPOINT"]= "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"]= "Perplexis"

# Initialize session state with default values
default_values = {
    'ai_role': None,
    'chat_memory': None,
    'llm_chain_conversation': None,
    'chat_response': None,
    'session_id': str(uuid.uuid4()),
    # 'client_remote_ip': get_remote_ip(),
    'llm': None,
    'llm_top_p': 0.50,
    'llm_openai_presence_penalty': 0.00,
    'llm_openai_frequency_penalty': 1.00,
    'llm_openai_max_tokens': 1024,
    'llm_ollama_repeat_penalty': 1.00,
    'llm_ollama_num_ctx': None,
    'llm_ollama_num_predict': None,
    'selected_embedding_provider': None,
    'selected_ai': None,
    'selected_llm': None,
    'temperature': 0.50,
    'metadata_history': [],
}

def init_session_state():
    """
    스트림릿 세션 상태를 초기화하는 함수입니다. 
    채팅 상태, 파라미터 등을 저장해 사용자의 대화 맥락을 유지합니다.
    """
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
        if key not in ['store', 'documents_chunks']:
            print(f"[DEBUG] (session_state) {key} = {st.session_state.get(key, 'Unknown')}")
    os.system('rm -rf ./uploads/*')

init_session_state()

# 추가: Streamlit 콜백 핸들러
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.placeholder.write(self.text)

#--------------------------------------------------

def main():
    """
    메인 함수로서, 스트림릿 인터페이스를 구성하고 
    AI 모델 파라미터 설정 및 채팅 기능을 제공합니다.
    """
    # 사이드바 영역 구성
    with st.sidebar:
        st.title("Parameters")

        st.session_state.ai_role = st.selectbox("Role of AI", get_ai_role_and_sysetm_prompt(only_key=True), index=0)

        st.session_state.selected_ai = st.radio("**:blue[AI]**", ("Ollama", "OpenAI"), index=0)
        

        if st.session_state.selected_ai == "OpenAI":
            os.environ["OPENAI_API_KEY"] = st.text_input("**:red[OpenAI API Key]** [Learn more](https://platform.openai.com/docs/quickstart)", value=os.environ["OPENAI_API_KEY"], type="password")
            os.environ["OPENAI_BASE_URL"] = st.text_input("OpenAI API URL", value=os.environ["OPENAI_BASE_URL"])

        if st.session_state.selected_embedding_provider == "Ollama" or st.session_state.selected_ai == "Ollama":
            os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama API URL", value=os.environ["OLLAMA_BASE_URL"])

        col_ai_llm, col_ai_temperature = st.sidebar.columns(2)
        with col_ai_llm:
            if st.session_state.selected_ai == "OpenAI":
                st.session_state.selected_llm = st.selectbox("AI LLM", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], index=4)
            else:
                st.session_state.selected_llm = st.selectbox("AI LLM", ["gemma2:2b", "gemma2:9b", "gemma2:27b", "call518/gemma2-uncensored-8192ctx:9b", "mistral:7b", "llama3.2:1b", "llama3:8b", "llama3.2:3b", "codegemma:2b", "codegemma:7b", "mistral:7b", "call518/deepseek-r1-32768ctx:8b", "call518/deepseek-r1-32768ctx:14b", "call518/EEVE-8192ctx:10.8b-q4", "call518/EEVE-8192ctx:10.8b-q5", "llama3-groq-tool-use:8b", "tulu3:8b", "call518/exaone3.5-32768ctx:2.4b", "call518/exaone3.5-32768ctx:7.8b", "call518/exaone3.5-32768ctx:32b"], index=0)
        with col_ai_temperature:
            # st.session_state['temperature'] = st.text_input("LLM Temperature (0.0 ~ 1.0)", value=st.session_state['temperature'])
            st.session_state.temperature = st.number_input("AI Temperature", min_value=0.00, max_value=1.00, value=st.session_state.temperature, step=0.05)

        ### LLM 모델 설정
        col_llm_top_p, col_repeat_penalty = st.sidebar.columns(2)
        with col_llm_top_p:
            st.session_state.llm_top_p = st.number_input("top_p", min_value=0.00, max_value=1.00, value=st.session_state.llm_top_p, step=0.05)
        with col_repeat_penalty:
            if st.session_state.selected_ai == "OpenAI":
                st.session_state.llm_openai_frequency_penalty = st.number_input("frequency_penalty", min_value=-2.00, max_value=2.00, value=st.session_state.llm_openai_frequency_penalty, step=0.05)
            if st.session_state.selected_ai == "Ollama":
                st.session_state.llm_ollama_repeat_penalty = st.number_input("repeat_penalty", min_value=0.00, value=st.session_state.llm_ollama_repeat_penalty, step=0.05)

        if st.session_state.selected_ai == "OpenAI":
            col_llm_openai_max_tokens, col_llm_openai_presence_penalty = st.sidebar.columns(2)
            with col_llm_openai_max_tokens:
                st.session_state.llm_openai_max_tokens = st.number_input("max_tokens", min_value=1024, max_value=get_max_value_of_model_max_tokens(st.session_state.selected_llm), value=get_max_value_of_model_max_tokens(st.session_state.selected_llm))
            with col_llm_openai_presence_penalty:
                st.session_state.llm_openai_presence_penalty = st.number_input("presence_penalty", min_value=-2.00, max_value=2.00, value=st.session_state.llm_openai_presence_penalty, step=0.05)
        if st.session_state.selected_ai == "Ollama":
            col_llm_ollama_num_ctx, col_llm_ollama_num_predict = st.sidebar.columns(2)
            with col_llm_ollama_num_ctx:
                st.session_state.llm_ollama_num_ctx = st.number_input("num_ctx", min_value=256, max_value=get_max_value_of_model_num_ctx(st.session_state.selected_llm), value=int(get_max_value_of_model_num_ctx(st.session_state.selected_llm) / 2), step=512)
                print(f"[DEBUG] (llm_ollama_num_ctx) {st.session_state.llm_ollama_num_ctx}")
            with col_llm_ollama_num_predict:
                st.session_state.llm_ollama_num_predict = st.number_input("num_predict", value=int(get_max_value_of_model_num_predict(st.session_state.selected_llm) / 2))
                print(f"[DEBUG] (llm_ollama_num_predict) {st.session_state.llm_ollama_num_predict}")

        ### Set OpenAI API Key
        if st.session_state.selected_ai == "OpenAI":
            if not os.environ["OPENAI_API_KEY"]:
                st.error("Please enter the OpenAI API Key.")
                st.stop()

        if st.button("Reset", type='primary'):
            streamlit_js_eval(js_expressions="parent.window.location.reload()")

#-----------------------------------------------------------------------------------------------------------

    # common_system_prompt = "최종 답변은 반드시 한국어로 작성해야 한다."
    common_system_prompt = "Write the answer in Korean if possible."
    st.session_state.system_prompt_content = get_ai_role_and_sysetm_prompt(st.session_state.ai_role) + " " + common_system_prompt
    
    ### Chat Memory 객체 생성 (최초 1회만 생성)
    if st.session_state.chat_memory is None:
        st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)

    ## Container 선언 순서가 화면에 보여지는 순서 결정
    container_history = st.container()
    container_user_textbox = st.container()

    # print(f"[DEBUG] (metadata_history) ==================> {st.session_state.selected_llm}")

    ### container_user_textbox 처리
    with container_user_textbox:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send', help="Click to send your message", icon=":material/send:", type='primary')

        if submit_button and user_input:
            streaming_placeholder = st.empty()  # 토큰 실시간 출력을 위한 컨테이너
            callback_handler = StreamlitCallbackHandler(placeholder=streaming_placeholder)

            # LLM 재생성: streaming=True 및 callbacks 적용
            if st.session_state.selected_ai == "OpenAI":
                st.session_state.llm = ChatOpenAI(
                    base_url = os.environ["OPENAI_BASE_URL"],
                    model = st.session_state.selected_llm,
                    temperature = st.session_state.temperature,
                    cache = False,
                    streaming = True,
                    callbacks = [callback_handler],
                    presence_penalty = st.session_state.llm_openai_presence_penalty,
                    frequency_penalty = st.session_state.llm_openai_frequency_penalty,
                    stream_usage = False,
                    n = 1,
                    top_p = st.session_state.llm_top_p,
                    max_tokens = st.session_state.llm_openai_max_tokens,
                    verbose = True,
                )
            else:
                st.session_state.llm = OllamaLLM(
                    base_url = os.environ["OLLAMA_BASE_URL"],
                    model = st.session_state.selected_llm,
                    temperature = st.session_state.temperature,
                    cache = False,
                    streaming = True,
                    callbacks = [callback_handler],
                    num_ctx = st.session_state.llm_ollama_num_ctx,
                    num_predict = st.session_state.llm_ollama_num_predict,
                    repeat_penalty = st.session_state.llm_ollama_repeat_penalty,
                    top_p = st.session_state.llm_top_p,
                    verbose = True,
                )

            # llm_chain_conversation 재생성 (새 llm 적용)
            st.session_state.llm_chain_conversation = LLMChain(
                llm=st.session_state.llm,
                prompt=ChatPromptTemplate(
                    messages=[
                        SystemMessagePromptTemplate.from_template(st.session_state.system_prompt_content),
                        MessagesPlaceholder(variable_name="chat_memory"),
                        HumanMessagePromptTemplate.from_template("{question}")
                    ]
                ),
                memory=st.session_state.chat_memory,
                verbose=True,
            )

            with st.spinner('Thinking...'):
                start_time = time.perf_counter()
                response = st.session_state.llm_chain_conversation.run({"question": user_input})
                st.session_state.chat_response = response
                end_time = time.perf_counter()
                # st.session_state.chat_history_elapsed_time.append(end_time - start_time)

            # metadata_history 추가
            metadata_entry = {
                "ai": st.session_state.selected_ai,
                "llm": st.session_state.selected_llm,
                "temperature": st.session_state.temperature,
                "llm_top_p": st.session_state.llm_top_p,
                "llm_openai_presence_penalty": st.session_state.llm_openai_presence_penalty,
                "llm_openai_frequency_penalty": st.session_state.llm_openai_frequency_penalty,
                "llm_openai_max_tokens": st.session_state.llm_openai_max_tokens,
                "llm_ollama_repeat_penalty": st.session_state.llm_ollama_repeat_penalty,
                "llm_ollama_num_ctx": st.session_state.llm_ollama_num_ctx,
                "llm_ollama_num_predict": st.session_state.llm_ollama_num_predict,
                "elapsed_time": end_time - start_time,
            }
            st.session_state.metadata_history.append(metadata_entry)

            metadata_entry_str = "\n".join([f"{key}: {value}" for key, value in metadata_entry.items() if key not in ('user', 'assistant')])
            full_message = "===== Perplexis:Chat =====\n\n[User]\n" + user_input + "\n\n[Assistant]\n" + response + "\n\n[Metadata]\n" + metadata_entry_str
            print(f"[DEBUG] (full_message) =======> {full_message}")
            try:
                asyncio.run(telegramSendMessage(full_message, os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")))
            except Exception as e:
                print(f"[ERROR] Telegram messaging failed: {e}")

            streaming_placeholder.empty()  # 추가: 스트리밍 출력 후 placeholder 비우기

    ### container_history 처리
    if st.session_state.chat_memory is not None:
        with container_history:
            caht_count = 0
            for message in st.session_state.chat_memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown("**<span style='color: blue;'>You</span>**", unsafe_allow_html=True)
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    metadata_history_entry = st.session_state.metadata_history[caht_count]
                    md_ai = metadata_history_entry["ai"]
                    md_llm = metadata_history_entry["llm"]
                    md_temperature = metadata_history_entry["temperature"]
                    md_llm_top_p = metadata_history_entry["llm_top_p"]
                    md_llm_openai_presence_penalty = metadata_history_entry["llm_openai_presence_penalty"]
                    md_llm_openai_frequency_penalty = metadata_history_entry["llm_openai_frequency_penalty"]
                    md_llm_openai_max_tokens = metadata_history_entry["llm_openai_max_tokens"]
                    md_llm_ollama_repeat_penalty = metadata_history_entry["llm_ollama_repeat_penalty"]
                    md_llm_ollama_num_ctx = metadata_history_entry["llm_ollama_num_ctx"]
                    md_llm_ollama_num_predict = metadata_history_entry["llm_ollama_num_predict"]
                    md_elapsed_time = metadata_history_entry["elapsed_time"]
                    with st.chat_message("assistant"):
                        st.markdown(f"**<span style='color: green;'>{st.session_state.ai_role}</span>**", unsafe_allow_html=True)
                        st.write(message.content)
                        with st.expander("Show Metadata"):
                            st.markdown(
                                f"""
                                <div style="color: #2E9AFE; border: 1px solid #ddd; padding: 5px; background-color: #f9f9f9; border-radius: 5px; width: 100%;">
                                    - <b>AI</b>: <font color=red>{md_ai}</font><br>
                                    - <b>LLM Model</b>: <font color=black>{md_llm}</font><br>
                                    - <b>LLM Args(Common) temperature</b>: <font color=black>{md_temperature}</font><br>
                                    - <b>LLM Args(Common) top_p</b>: <font color=black>{md_llm_top_p}</font><br>
                                    - <b>LLM Args(OpenAI) presence_penalty</b>: <font color=black>{md_llm_openai_presence_penalty}</font><br>
                                    - <b>LLM Args(OpenAI) frequency_penalty</b>: <font color=black>{md_llm_openai_frequency_penalty}</font><br>
                                    - <b>LLM Args(OpenAI) max_tokens</b>: <font color=black>{md_llm_openai_max_tokens}</font><br>
                                    - <b>LLM Args(Ollama) repeat_penalty</b>: <font color=black>{md_llm_ollama_repeat_penalty}</font><br>
                                    - <b>LLM Args(Ollama) num_ctx</b>: <font color=black>{md_llm_ollama_num_ctx}</font><br>
                                    - <b>LLM Args(Ollama) num_predict</b>: <font color=black>{md_llm_ollama_num_predict}</font><br>
                                    - <b>Elapsed time</b>: <font color=black>{md_elapsed_time:.0f} sec</font><br>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.write("")
                    caht_count += 1

#----------------------------------------------------

if __name__ == "__main__":
    Navbar()
    
    main()
    # # Error 메세지/코드 숨기기
    # try:
    #     main()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
