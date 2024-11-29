# pip install -U langchain-community bs4 langchain_pinecone pinecone-client[grpc] langchain-openai langchain_ollama streamlit-chat streamlit-js-eval googlesearch-python chromadb pysqlite3-binary pypdf pymupdf rapidocr-onnxruntime langchain-experimental
# pip list --format=freeze > requirements.txt (또는 pip freeze > requirements.txt)

### (임시) pysqlite3 설정 - sqlite3 모듈을 pysqlite3로 대체
### pip install pysqlite3-binary 필요
### "Your system has an unsupported version of sql requires sqlite3 >= 3.35.0." 오류 해결 목적
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from modules.nav import Navbar
from modules.common_functions import get_ai_role_and_sysetm_prompt
from modules.common_functions import get_country_name_by_code
from modules.common_functions import get_max_value_of_model_max_tokens
from modules.common_functions import get_max_value_of_model_num_ctx
from modules.common_functions import get_max_value_of_model_embedding_dimensions

import streamlit as st
# from streamlit_chat import message
from streamlit_js_eval import streamlit_js_eval
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_ollama import OllamaLLM

# from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings ### (임시) OllamaEmbeddings 모듈 임포트 수정 (langchain_ollama 는 임베딩 실패 발생)

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
import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from googlesearch import search

import os
import shutil
import requests

# 페이지 정보 정의
st.set_page_config(page_title="Perplexis:Chat", page_icon=":books:", layout="wide")
st.title(":books: _:red[Perplexis]_ Chat")

### Mandatory Keys 설정
### 1. secretes.toml 파일에 설정된 KEY 값이 최우선 적용.
### 2. secretes.toml 파일에 설정된 KEY 값이 없을 경우, os.environ 환경변수로 설정된 KEY 값이 적용.
### 3. secretes.toml 파일과 os.environ 환경변수 모두 설정되지 않은 경우, Default 값을 적용.
if st.secrets["KEYS"].get("OLLAMA_BASE_URL"):
    os.environ["OLLAMA_BASE_URL"] = st.secrets["KEYS"].get("OLLAMA_BASE_URL")
elif not os.environ.get("OLLAMA_BASE_URL"):
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

if st.secrets["KEYS"].get("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = st.secrets["KEYS"].get("OPENAI_BASE_URL")
elif not os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"

if st.secrets["KEYS"].get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["KEYS"].get("OPENAI_API_KEY")
elif not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = ""

if st.secrets["KEYS"].get("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = st.secrets["KEYS"].get("PINECONE_API_KEY")
elif not os.environ.get("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = ""

### (Optional) Langchain API Key 설정
if st.secrets["KEYS"].get("LANGCHAIN_API_KEY", ""):
    os.environ["LANGCHAIN_TRACING_V2"]= "true"
    os.environ["LANGCHAIN_ENDPOINT"]= "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["KEYS"].get("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"]= "Perplexis"

# (임시) Session ID값으로 Client IP를 사용 (추후 ID/PW 시스템으로 변경 필요)
def get_remote_ip() -> str:
    """Get remote ip."""

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            st.error(f"[ERROR] Client IP 확인 오류 발생: ctx is None")
            st.stop()

        session_info = runtime.get_instance().get_client(ctx.session_id)
        print(session_info.request)
        if session_info is None:
            st.error(f"[ERROR] Client IP 확인 오류 발생: session_info is None")
            st.stop()
    except Exception as e:
        st.error(f"[ERROR] Client IP 확인 오류 발생: {e}")
        st.stop()

    return session_info.request.remote_ip

# Initialize session state with default values
default_values = {
    'ai_role': None,
    'system_prompt_content': None,
    'chat_memory': None,
    'chat_conversation': None,
    'chat_response': None,
    'session_id': str(uuid.uuid4()),
    'client_remote_ip' : get_remote_ip(),
    'is_analyzed': False,
    'document_type': None,
    'document_source': [],
    'documents_chunks': [],
    'embedding_instance': None,
    'llm': None,
    'llm_top_p': 0.80,
    'llm_openai_presence_penalty': 0.00,
    'llm_openai_frequency_penalty': 1.00,
    'llm_openai_max_tokens': 1024,
    'llm_openai_max_tokens_fix': 1024,
    'llm_ollama_repeat_penalty': 1.00,
    'llm_ollama_num_ctx': 1024,
    'llm_ollama_num_ctx_fix': 1024,
    'llm_ollama_num_predict': -1,
    'selected_embedding_provider': None,
    'selected_embedding_model': None,
    'selected_embedding_dimension': None,
    'vectorstore_type': "pinecone",
    'selected_ai': None,
    'selected_llm': None,
    'temperature': 0.00,
    'chunk_size': None,
    'chunk_overlap': None,
    'retriever': None,
    'pinecone_index_reset': True,
    'chromadb_root_reset': True,
    'rag_search_type': None,
    'rag_score': 0.01,
    'rag_top_k': None,
    'rag_fetch_k': None,
    'rag_lambda_mult': None,
    'google_search_query': None,
    'google_custom_urls': None,
    'google_search_result_count': None,
    'google_search_doc_lang': None,
    'rag_history_user': [],
    'rag_history_ai': [],
    'rag_history_llm_model_name': [],
    'rag_history_rag_contexts': [],
    'rag_history_temperature': [],
    'rag_history_rag_search_type': [],
    'rag_history_rag_top_k': [],
    'rag_history_rag_score': [],
    'rag_history_rag_fetch_k': [],
    'rag_history_rag_rag_lambda_mult': [],
    'store': {},
}

def init_session_state():
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
        if key not in ['store', 'documents_chunks']:
            print(f"[DEBUG] (session_state) {key} = {st.session_state.get(key, 'Unknown')}")
    os.system('rm -rf ./uploads/*')

init_session_state()
upload_dir = f"./uploads/{st.session_state['session_id']}"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# URL 패턴 정의 (기본적인 URL 형태를 검증)
url_pattern = re.compile(
    r'^(?:http|ftp)s?://'  # http:// 또는 https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # 도메인 이름
    r'localhost|'  # localhost
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IPv4 주소
    r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # IPv6 주소
    r'(?::\d+)?'  # 포트 번호
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

### Google Search 처리, URL 목록 반환
def google_search(query, num_results=10, lang="ko"):
    results_list = []
    try:
        if lang == "Any":
            results = search(query, num_results=num_results)
        else:
            results = search(query, num_results=num_results, lang=lang)
        
        if results:
            for idx, result in enumerate(results, 1):
                # PDF 링크 제외
                try:
                    response = requests.head(result, allow_redirects=True)
                    if 'application/pdf' not in response.headers.get('Content-Type', ''):
                        results_list.append(result)
                        print(f"[DEBUG] (Google Search URLs) {idx}. {result}")
                    else:
                        print(f"[DEBUG] (Google Search URLs) {idx}. <PDF Removed> {result}")
                except Exception as e:
                    print(f"[ERROR] Failed to check URL {result}: {e}")
            return results_list
        else:
            st.error("No search results found.")
            st.stop
    except Exception as e:
        st.error(f"[ERROR] {e}")
        st.stop()

### AI 모델 정보 반환
def get_llm_model_name():
    # AI 모델 정보 표시
    ai_model = st.session_state.get('llm', None)
    if ai_model:
        llm_model_name = ai_model.model_name if hasattr(ai_model, 'model_name') else ai_model.model
    else:
        llm_model_name = "Unknown LLM"
    return llm_model_name

### ChromaDB 초기화 정의
chromadb_root = './chromadb'
def reset_chromadb(db_path=chromadb_root):
    ### ChromaDB 논리 세션 초기화
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    
    ### ChromaDB 물리 세션 초기화 (DB 삭제)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path)

### URL 유효성 검사
def is_valid_url(url):
    return re.match(url_pattern, url) is not None

# Pinecone 관련 설정
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# pinecone_index_name = 'perplexis'

### 파일 업로드 후, 업로드된 파일 프로세싱
def read_txt(file):
    # upload_dir = f"./uploads/{st.session_state['session_id']}"
    loader = TextLoader(f"{upload_dir}/{file}")
    docs = loader.load()
    # shutil.rmtree(upload_dir)
    return docs

### 파일 업로드 후, 업로드된 파일 프로세싱
def read_pdf(file):
    ### pypdf 모듈을 이용한 PDF 파일 로드
    # loader = PyPDFLoader(f"{upload_dir}/{file}")
    ### pymupdf 모듈을 이용한 PDF 파일 로드
    loader = PyMuPDFLoader(f"{upload_dir}/{file}", extract_images=True)
    docs = loader.load()
    # shutil.rmtree(upload_dir)
    return docs

#--------------------------------------------------

def main():
    # 사이드바 영역 구성
    with st.sidebar:
        st.title("Parameters")

        st.session_state['ai_role'] = st.selectbox("Role of AI", get_ai_role_and_sysetm_prompt(only_key=True), index=0)

        st.session_state['selected_ai'] = st.radio("**:blue[AI]**", ("Ollama", "OpenAI"), index=0, disabled=st.session_state['is_analyzed'])
        

        if st.session_state.get('selected_ai', "Ollama") == "OpenAI":
            os.environ["OPENAI_API_KEY"] = st.text_input("**:red[OpenAI API Key]** [Learn more](https://platform.openai.com/docs/quickstart)", value=os.environ["OPENAI_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
            os.environ["OPENAI_BASE_URL"] = st.text_input("OpenAI API URL", value=os.environ["OPENAI_BASE_URL"], disabled=st.session_state['is_analyzed'])

        if st.session_state.get('selected_embedding_provider', "Ollama") == "Ollama" or st.session_state.get('selected_ai', "Ollama") == "Ollama":
            os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama API URL", value=os.environ["OLLAMA_BASE_URL"], disabled=st.session_state['is_analyzed'])

        col_ai_llm, col_ai_temperature = st.sidebar.columns(2)
        with col_ai_llm:
            if st.session_state['selected_ai'] == "OpenAI":
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], index=4, disabled=st.session_state['is_analyzed'])
            else:
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gemma2:2b", "gemma2:9b", "gemma2:27b", "mistral:7b", "llama3.2:1b", "llama3.2:3b", "codegemma:2b", "codegemma:7b"], index=1, disabled=st.session_state['is_analyzed'])
        with col_ai_temperature:
            # st.session_state['temperature'] = st.text_input("LLM Temperature (0.0 ~ 1.0)", value=st.session_state['temperature'])
            st.session_state['temperature'] = st.number_input("AI Temperature", min_value=0.00, max_value=1.00, value=st.session_state['temperature'], step=0.05, disabled=st.session_state['is_analyzed'])

        ### LLM 모델 설정
        col_llm_top_p, col_repeat_penalty = st.sidebar.columns(2)
        with col_llm_top_p:
            st.session_state['llm_top_p'] = st.number_input("top_p", min_value=0.00, max_value=1.00, value=st.session_state.get('llm_top_p', 0.80), step=0.05, disabled=st.session_state['is_analyzed'])
        with col_repeat_penalty:
            if st.session_state.get('selected_ai', "Ollama") == "OpenAI":
                st.session_state['llm_openai_frequency_penalty'] = st.number_input("frequency_penalty", min_value=-2.00, max_value=2.00, value=st.session_state.get('llm_openai_frequency_penalty', 1.00), step=0.05, disabled=st.session_state['is_analyzed'])
            if st.session_state.get('selected_ai', "Ollama") == "Ollama":
                st.session_state['llm_ollama_repeat_penalty'] = st.number_input("repeat_penalty", min_value=0.00, value=st.session_state.get('llm_ollama_repeat_penalty', 1.10), step=0.05, disabled=st.session_state['is_analyzed'])

        if st.session_state.get('selected_ai', "Ollama") == "OpenAI":
            col_llm_openai_max_tokens, col_llm_openai_presence_penalty = st.sidebar.columns(2)
            with col_llm_openai_max_tokens:
                st.session_state['llm_openai_max_tokens'] = st.number_input("max_tokens", min_value=1024, max_value=get_max_value_of_model_max_tokens(st.session_state['selected_llm']), value=get_max_value_of_model_max_tokens(st.session_state['selected_llm']), disabled=st.session_state['is_analyzed'])
            with col_llm_openai_presence_penalty:
                st.session_state['llm_openai_presence_penalty'] = st.number_input("presence_penalty", min_value=-2.00, max_value=2.00, value=st.session_state.get('llm_openai_presence_penalty', 1.00), step=0.05, disabled=st.session_state['is_analyzed'])
        if st.session_state.get('selected_ai', "Ollama") == "Ollama":
            col_llm_ollama_num_ctx, col_llm_ollama_num_predict = st.sidebar.columns(2)
            with col_llm_ollama_num_ctx:
                st.session_state['llm_ollama_num_ctx'] = st.number_input("num_ctx", min_value=1024, max_value=get_max_value_of_model_num_ctx(st.session_state['selected_llm']), value=get_max_value_of_model_num_ctx(st.session_state['selected_llm']), disabled=st.session_state['is_analyzed'])
            with col_llm_ollama_num_predict:
                st.session_state['llm_ollama_num_predict'] = st.number_input("num_predict", value=st.session_state.get('llm_ollama_num_predict', -1), disabled=st.session_state['is_analyzed'])

        ### Set OpenAI API Key
        if st.session_state.get('selected_ai', "Ollama") == "OpenAI":
            if not os.environ["OPENAI_API_KEY"]:
                st.error("Please enter the OpenAI API Key.")
                st.stop()

        # AI 모델 선택 및 초기화
        if st.session_state['selected_ai'] == "OpenAI":
            st.session_state['llm'] = ChatOpenAI(
                base_url = os.environ["OPENAI_BASE_URL"],
                model = st.session_state.get('selected_llm', "gpt-3.5-turbo"),
                temperature = st.session_state.get('temperature', 0.00),
                cache = False,
                streaming = False,
                presence_penalty = st.session_state.get('llm_openai_presence_penalty', 1.00),
                frequency_penalty = st.session_state.get('llm_openai_frequency_penalty', 1.00),
                stream_usage = False,
                n = 1,
                top_p = st.session_state.get('llm_top_p', 0.50),
                max_tokens = st.session_state.get('llm_openai_max_tokens', 1024),
                verbose = True,
            )
        else:
            st.session_state['llm'] = OllamaLLM(
                base_url = os.environ["OLLAMA_BASE_URL"],
                model = st.session_state.get('selected_llm', "gemma2:9b"),
                temperature = st.session_state.get('temperature', 0.00),
                cache = False,
                num_ctx = st.session_state.get('llm_ollama_num_ctx', 1024),
                num_predict = st.session_state.get('llm_ollama_num_predict', -1),
                # num_gpu = None,
                # num_thread = None,
                # repeat_last_n = None,
                repeat_penalty = st.session_state.get('llm_ollama_repeat_penalty', 1.00),
                # tfs_z = None,
                # top_k = None,
                top_p = st.session_state.get('llm_top_p', 0.80),
                # format = "", # Literal['', 'json'] (default: "")
                verbose = True,
            )

        if st.session_state.get('selected_mode', "Chat") == "Chat":
            if st.button("Reset", type='primary'):
                streamlit_js_eval(js_expressions="parent.window.location.reload()")

#-----------------------------------------------------------------------------------------------------------

    # system_prompt_content = "You are a chatbot having a conversation with a human."
    # system_prompt_content = "I want you to act as an academician. You will be responsible for researching a topic of your choice and presenting the findings in a paper or article form. Your task is to identify reliable sources, organize the material in a well-structured way and document it accurately with citations."
    st.session_state['system_prompt_content'] = get_ai_role_and_sysetm_prompt(st.session_state.get('ai_role', "Basic chatbot"))
    
    system_prompt = st.session_state['system_prompt_content']

    ### Chat Memory 객체 생성 (최초 1회만 생성)
    if st.session_state.get('chat_memory', None) is None:
        st.session_state['chat_memory'] = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)

    ### Chat Prompt 객체 생성
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_memory"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    ### Chat Conversation 객체 생성        
    st.session_state['chat_conversation'] = LLMChain(
        llm=st.session_state['llm'],
        prompt=prompt,
        memory=st.session_state['chat_memory'],
        verbose=True,
    )        

    ## Container 선언 순서가 화면에 보여지는 순서 결정
    container_history = st.container()
    container_user_textbox = st.container()

    ### container_user_textbox 처리
    with container_user_textbox:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send', use_container_width=False, help="Click to send your message", on_click=None, args=None, kwargs=None, disabled=False, icon=":material/send:", type='primary')
            
        if submit_button and user_input:
            with st.spinner('Thinking...'):
                st.session_state['chat_response'] = st.session_state['chat_conversation'].invoke({"question": user_input})
                # print(f"[DEBUG] (chat_response) {st.session_state['chat_response']}")

    ### container_history 처리
    if st.session_state.get('chat_memory', None) is not None:
        with container_history:
            for message in st.session_state['chat_memory'].chat_memory.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown("**<span style='color: blue;'>You</span>**", unsafe_allow_html=True)
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(f"**<span style='color: green;'>{st.session_state.get('ai_role', 'Unknown')}</span>**", unsafe_allow_html=True)
                        st.write(message.content)

#----------------------------------------------------

if __name__ == "__main__":
    Navbar()
    
    main()
    # # Error 메세지/코드 숨기기
    # try:
    #     main()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
