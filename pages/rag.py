# pip install -U langchain-community bs4 langchain_pinecone pinecone-client[grpc] langchain-openai langchain_ollama streamlit-chat streamlit-js-eval googlesearch-python chromadb pysqlite3-binary pypdf pymupdf rapidocr-onnxruntime langchain-experimental langchain_postgres psycopg_binary sqlalchemy
# pip list --format=freeze > requirements.txt
# sed -i '/^pip==/d' requirements.txt
# (Manual Run PGVector Docker) docker run --name pgsql-pgvectror-perplexis -e POSTGRES_USER=perplexis -e POSTGRES_PASSWORD=changeme -e POSTGRES_DB=perplexis -p 5432:5432 -d -v pgvector-pgdata:/var/lib/postgresql/data call518/pgvector:pg16-1.0.0

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

#--------------------------------------------------

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

### (임시) OllamaEmbeddings 모듈 임포트 수정 (langchain_ollama 는 임베딩 실패 발생)
# from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

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
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

#--------------------------------------------------

# 페이지 정보 정의
st.set_page_config(page_title="Perplexis:Rag", page_icon=":books:", layout="wide")
st.title(":books: _:red[Perplexis]_ RAG")

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

if st.secrets["KEYS"].get("PGVECTOR_HOST"):
    os.environ["PGVECTOR_HOST"] = st.secrets["KEYS"].get("PGVECTOR_HOST")
elif not os.environ.get("PGVECTOR_HOST"):
    os.environ["PGVECTOR_HOST"] = "localhost"

if st.secrets["KEYS"].get("PGVECTOR_PORT"):
    os.environ["PGVECTOR_PORT"] = st.secrets["KEYS"].get("PGVECTOR_PORT")
elif not os.environ.get("PGVECTOR_PORT"):
    os.environ["PGVECTOR_PORT"] = "5432"

if st.secrets["KEYS"].get("PGVECTOR_USER"):
    os.environ["PGVECTOR_USER"] = st.secrets["KEYS"].get("PGVECTOR_USER")
elif not os.environ.get("PGVECTOR_USER"):
    os.environ["PGVECTOR_USER"] = "perplexis"

if st.secrets["KEYS"].get("PGVECTOR_PASS"):
    os.environ["PGVECTOR_PASS"] = st.secrets["KEYS"].get("PGVECTOR_PASS")
elif not os.environ.get("PGVECTOR_PASS"):
    os.environ["PGVECTOR_PASS"] = "changeme"

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
    'vectorstore_type': None,
    'pgvector_connection': None,
    'pinecone_similarity': None,
    'chromadb_similarity': None,
    'pgvector_similarity': None,
    'selected_ai': None,
    'selected_llm': None,
    'temperature': 0.00,
    'chunk_size': None,
    'chunk_overlap': None,
    'retriever': None,
    'pinecone_index_reset': True,
    'chromadb_root_reset': True,
    'pgvector_db_reset': True,
    'rag_search_type': None,
    'rag_score_threshold': 0.50,
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
    'rag_history_rag_score_threshold': [],
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

def reset_pgvector(connection):
    # Create a new SQLAlchemy engine
    engine = create_engine(connection)

    # Create a new session
    Session = sessionmaker(bind=engine)
    session = Session()

    tables = [ "langchain_pg_embedding", "langchain_pg_collection" ]

    from sqlalchemy import text

    for table in tables:
        # Drop the existing collection if it exists
        session.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    
    session.commit()
    session.close()

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

        st.session_state['ai_role'] = st.selectbox("Role of AI", get_ai_role_and_sysetm_prompt(only_key=True), index=2)

        # Contextualize question (질답 히스토리를 이용해, 주어진 질문을 문맥상 정확하고 독립적인 질문으로 보완/재구성 처리해서 반환)
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
            # "채팅 기록과 사용자가 마지막으로 한 질문이 주어졌을 때, 해당 질문이 채팅 기록의 맥락을 참조할 수 있으므로, 채팅 기록 없이도 이해할 수 있는 독립적인 질문으로 다시 작성하세요. 질문에 답변하지 말고, 필요하다면 질문을 재구성하고, 그렇지 않다면 그대로 반환하세요."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # QA 시스템 프롬프트 설정 (사용자 질문과, 벡터DB로부터 얻은 결과를 조합해 최종 답변을 작성하기 위한 행동 지침 및 질답 체인 생성)
        system_prompt_content = get_ai_role_and_sysetm_prompt(st.session_state['ai_role'])
        system_prompt = (
            # "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."
            # "당신은 모든 분야에서 전문가 입니다. 다음 검색된 컨텍스트 조각을 사용하여 질문자의 질문에 체계적인 답변을 작성하세요. 답을 모르면 모른다고 말하세요. 최소 50개의 문장을 사용하고 답변은 간결하면서도 최대한 상세한 정보를 포함해야 합니다. (답변은 한국어로 작성하세요.)"
            # "You are an expert in all fields. Using the following retrieved context snippets, formulate a systematic answer to the asker's question. If you don't know the answer, say you don't know. Use at least 50 sentences, and ensure the response is concise yet contains as much detailed information as possible. (The response should be written in Korean.)"
            # "You are an expert in all fields. Using the context document fragments provided through RAG, compose systematic answers that correspond to the questioner's questions. If you don't know the answer, say you don't know. Use at least 50 sentences, and the answer should be concise yet include as much detailed information as possible. (Write the answer in Korean.)"
            # "You are an expert in all fields. Using the context document fragments provided through RAG, compose systematic answers that correspond to the questioner's questions. If you don't know the answer, say you don't know. Use at least 50 sentences, and the answer should be concise yet include as much detailed information as possible. (Write the answer in Korean.)"
            f"{system_prompt_content}"
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        st.session_state['document_type'] = st.radio("Document Type", ("URL", "File Upload", "Google Search"), horizontal=True, disabled=st.session_state['is_analyzed'])

        if st.session_state['document_type'] == "URL":
            st.session_state['document_source'] = [ st.text_input("Document URL", disabled=st.session_state['is_analyzed']) ]
        elif st.session_state['document_type'] == "File Upload":
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                with open(f"{upload_dir}/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.write(f"File uploaded: {uploaded_file.name}")
                st.session_state['document_source'].append(uploaded_file.name)
        elif st.session_state['document_type'] == "Google Search":
            st.session_state['google_search_query'] = st.text_input("Google Search Query", placeholder="Enter Your Keywords", disabled=st.session_state['is_analyzed'])
            st.session_state['google_custom_urls'] = st.text_area("Google Search Custom URLs (per Line)", placeholder="Enter Your Keywords", disabled=st.session_state['is_analyzed'])
            col_google_search_result_count, col_google_search_doc_lang = st.sidebar.columns(2)
            with col_google_search_result_count:
                st.session_state['google_search_result_count'] = st.number_input("Search Results", min_value=1, max_value=50, value=10, step=1, disabled=st.session_state['is_analyzed'])
            with col_google_search_doc_lang:
                st.session_state['google_search_doc_lang'] = st.selectbox("Search Document Language", [ "Any", "en", "ko", "jp", "cn"], disabled=st.session_state['is_analyzed'])
        else:
            st.error("[ERROR] Unsupported document type")
            st.stop()

        col_ai, col_embedding, col_vectorstore = st.sidebar.columns(3)
        with col_ai:
            st.session_state['selected_ai'] = st.radio("**:blue[AI]**", ("Ollama", "OpenAI"), index=0, disabled=st.session_state['is_analyzed'])
        with col_embedding:
            st.session_state['selected_embedding_provider'] = st.radio("**:blue[Embeddings]**", ("Ollama", "OpenAI"), index=0, disabled=st.session_state['is_analyzed'])
        with col_vectorstore:
            st.session_state['vectorstore_type'] = st.radio("**:blue[VectorDB]**", ("PGVector", "ChromaDB", "Pinecone"), index=1, disabled=st.session_state['is_analyzed'])
        

        if st.session_state.get('selected_embedding_provider', "Ollama") == "OpenAI" or st.session_state.get('selected_ai', "Ollama") == "OpenAI":
            os.environ["OPENAI_API_KEY"] = st.text_input("**:red[OpenAI API Key]** [Learn more](https://platform.openai.com/docs/quickstart)", value=os.environ["OPENAI_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
            os.environ["OPENAI_BASE_URL"] = st.text_input("OpenAI API URL", value=os.environ["OPENAI_BASE_URL"], disabled=st.session_state['is_analyzed'])
    
        if st.session_state.get('selected_embedding_provider', "Ollama") == "Ollama" or st.session_state.get('selected_ai', "Ollama") == "Ollama":
            os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama API URL", value=os.environ["OLLAMA_BASE_URL"], disabled=st.session_state['is_analyzed'])
    
        if st.session_state['vectorstore_type'] == "ChromaDB":
            st.session_state['chromadb_root_reset'] = st.checkbox("Reset ChromaDB", value=st.session_state.get('chromadb_root_reset', True), disabled=st.session_state['is_analyzed'])
            st.session_state['chromadb_similarity'] = st.selectbox("ChromaDB Similarity", ["cosine", "l2", "ip"], index=0, disabled=st.session_state['is_analyzed'])


        if st.session_state['vectorstore_type'] == "PGVector":
            st.session_state['pgvector_db_reset'] = st.checkbox("Reset PGVector", value=st.session_state.get('pgvector_db_reset', True), disabled=st.session_state['is_analyzed'])

            if st.session_state['pgvector_connection'] is None:
                st.session_state['pgvector_connection'] = f"postgresql+psycopg://{os.environ['PGVECTOR_USER']}:{os.environ['PGVECTOR_PASS']}@{os.environ['PGVECTOR_HOST']}:{os.environ['PGVECTOR_PORT']}/perplexis"
            st.session_state['pgvector_connection'] = st.text_input("PGVector Connection", value=st.session_state['pgvector_connection'], type="password",  disabled=st.session_state['is_analyzed'])
            st.session_state['pgvector_similarity'] = st.selectbox("PGVector Similarity", ["cosine", "euclidean", "dotproduct"], index=0, disabled=st.session_state['is_analyzed'])
    
        if st.session_state['vectorstore_type'] == "Pinecone":
            os.environ["PINECONE_API_KEY"] = st.text_input("**:red[Pinecone API Key]** [Learn more](https://www.pinecone.io/docs/quickstart/)", value=os.environ["PINECONE_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
            st.session_state['pinecone_index_reset'] = st.checkbox("Reset Pinecone Index", value=st.session_state.get('pinecone_index_reset', False), disabled=st.session_state['is_analyzed'])
            st.session_state['pinecone_similarity'] = st.selectbox("Pinecone Similarity", ["cosine", "l2", "inner"], index=0, disabled=st.session_state['is_analyzed'])

        ### Embeddings 모델 선택/설정
        if st.session_state['selected_embedding_provider'] == "OpenAI":
            # text-embedding-3-small: dimension=1536 (Increased performance over 2nd generation ada embedding model)
            # text-embedding-3-large: dimension=3072 (Most capable embedding model for both english and non-english tasks)
            # text-embedding-ada-002: dimension=1536 (Most capable 2nd generation embedding model, replacing 16 first generation models)
            st.session_state['selected_embedding_model'] = st.selectbox("Embedding Model", ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"], index=0,  disabled=st.session_state['is_analyzed'])
        else:
            # mxbai-embed-large: dimension=1024 (State-of-the-art large embedding model from mixedbread.ai) ("num_ctx": 512)
            # nomic-embed-text: dimension=768 (A high-performing open embedding model with a large token context window.) ("num_ctx": 8192)
            # all-minilm : dimension=384 (Embedding models on very large sentence level datasets.) ("num_ctx": 256)
            st.session_state['selected_embedding_model'] = st.selectbox("Embedding Model", ["bge-m3:567m", "all-minilm:22m", "all-minilm:33m", "nomic-embed-text", "mxbai-embed-large", "llama3:8b"], index=0,  disabled=st.session_state['is_analyzed'])

        st.session_state['selected_embedding_dimension'] = get_max_value_of_model_embedding_dimensions(st.session_state.get('selected_embedding_model', None))
        print(f"[DEBUG] (selected_embedding_model) {st.session_state.get('selected_embedding_model', None)}")
        print(f"[DEBUG] (selected_embedding_dimension) {st.session_state.get('selected_embedding_dimension', None)}")
        if st.session_state['selected_embedding_dimension'] is None:
            st.error("[ERROR] Unsupported embedding model")
            st.stop()

        col_ai_llm, col_ai_temperature = st.sidebar.columns(2)
        with col_ai_llm:
            if st.session_state['selected_ai'] == "OpenAI":
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], index=4, disabled=st.session_state['is_analyzed'])
            else:
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gemma2:2b", "gemma2:9b", "gemma2:27b", "mistral:7b", "llama3.2:1b", "llama3.2:3b", "codegemma:2b", "codegemma:7b"], index=1, disabled=st.session_state['is_analyzed'])
        with col_ai_temperature:
            # st.session_state['temperature'] = st.text_input("LLM Temperature (0.0 ~ 1.0)", value=st.session_state['temperature'])
            st.session_state['temperature'] = st.number_input("AI Temperature", min_value=0.00, max_value=1.00, value=st.session_state['temperature'], step=0.05, disabled=st.session_state['is_analyzed'])

        col_chunk_size, col_chunk_overlap = st.sidebar.columns(2)
        with col_chunk_size:
            st.session_state['chunk_size'] = st.number_input("Chunk Size", min_value=100, max_value=5000, value=1000, step=100, disabled=st.session_state['is_analyzed'])
        with col_chunk_overlap:
            st.session_state['chunk_overlap'] = st.number_input("Chunk Overlap", min_value=50, max_value=1000, value=100, step=100, disabled=st.session_state['is_analyzed'])
            if not (st.session_state['chunk_size'] >= (st.session_state['chunk_overlap'] * 2)):
                st.error("Chunk Overlap must be less than Chunk Size.")
                st.stop()

        ### (임시) similarity_score_threshold 모드에서, score_threshold 값 사용 안되는 버그 있음. (해결될 때 까지 노출 안함...)
        st.session_state['rag_search_type'] = st.selectbox("RAG Search Type", ["similarity", "similarity_score_threshold", "mmr"], index=1, disabled=st.session_state['is_analyzed'])
        # st.session_state['rag_search_type'] = st.selectbox("RAG Search Type", ["similarity", "mmr"], index=0, disabled=st.session_state['is_analyzed'])
    
        if st.session_state['rag_search_type'] == "similarity_score_threshold":
            col_rag_arg1, col_rag_arg2 = st.sidebar.columns(2)
            with col_rag_arg1:
                st.session_state['rag_top_k'] = st.number_input("RAG Top-K", min_value=1, max_value=50, value=10, step=1, disabled=st.session_state['is_analyzed'])        
            with col_rag_arg2:
                st.session_state['rag_score_threshold'] = st.number_input("score_threshold", min_value=0.01, max_value=1.00, value=0.50, step=0.05, disabled=st.session_state['is_analyzed'])
        elif st.session_state['rag_search_type'] == "similarity":
            st.session_state['rag_top_k'] = st.number_input("RAG Top-K", min_value=1, max_value=50, value=10, step=1, disabled=st.session_state['is_analyzed'])
        elif st.session_state['rag_search_type'] == "mmr":
            col_rag_arg1, col_rag_arg2, col_rag_arg3 = st.sidebar.columns(3)
            with col_rag_arg1:
                st.session_state['rag_top_k'] = st.number_input("RAG Top-K", min_value=1, max_value=50, value=10, step=1, disabled=st.session_state['is_analyzed'])
            with col_rag_arg2:
                st.session_state['rag_fetch_k'] = st.number_input("MMR Fetch-K", min_value=1, value=5, step=1, disabled=st.session_state['is_analyzed'])
            with col_rag_arg3:
                st.session_state['rag_lambda_mult'] = st.number_input("MMR Lambda Mult", min_value=0.01, max_value=1.00, value=0.50, step=0.05, disabled=st.session_state['is_analyzed'])

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
        if st.session_state.get('selected_embedding_provider', "Ollama") == "OpenAI" or st.session_state.get('selected_ai', "Ollama") == "OpenAI":
            if not os.environ["OPENAI_API_KEY"]:
                st.error("Please enter the OpenAI API Key.")
                st.stop()

        if st.session_state.get('vectorstore_type', 'Pinecone') == "Pinecone":
            if not os.environ["PINECONE_API_KEY"]:
                st.error("Please enter the Pinecone API Key.")
                st.stop()

        # Embedding 선택 및 초기화
        if st.session_state['selected_embedding_provider'] == "OpenAI":
            st.session_state['embedding_instance'] = OpenAIEmbeddings(
                base_url = os.environ["OPENAI_BASE_URL"],
                model = st.session_state.get('selected_embedding_model', None),
                # dimensions = st.session_state['selected_embedding_dimension'],
            )
        else:
            st.session_state['embedding_instance'] = OllamaEmbeddings(
                base_url = os.environ["OLLAMA_BASE_URL"],
                model = st.session_state.get('selected_embedding_model', None),
                # dimensions = st.session_state['selected_embedding_dimension'],
            )

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

        # 사용자 선택 및 입력값을 기본으로 RAG 데이터 준비
        if st.button("Embedding", disabled=st.session_state['is_analyzed']):
            docs_contents = []

            with st.spinner('Embedding...'):
                if st.session_state['document_type'] == "URL":
                    # 주요 세션 정보 구성
                    if not st.session_state['document_source']:
                        st.error("[ERROR] URL information is missing.")
                        st.stop()

                    if not is_valid_url(st.session_state['document_source'][0]):
                        st.error("[ERROR] Invalid URL.")
                        st.stop()
                    # 문서 로드 및 분할
                    loader = WebBaseLoader(
                        web_paths = (st.session_state['document_source'][0],),
                        # bs_kwargs=dict(
                        #     parse_only=bs4.SoupStrainer(
                        #         class_=("post-content", "post-title", "post-header")
                        #     )
                        # ),
                        show_progress = True,
                    )
                    docs_contents = loader.load()
                
                if st.session_state['document_type'] == "File Upload":
                    if uploaded_file is not None:
                        if uploaded_file.type == "application/pdf":
                            docs_contents = read_pdf(uploaded_file.name)
                        elif uploaded_file.type == "text/plain":
                            docs_contents = read_txt(uploaded_file.name)
                        else:
                            st.error("[ERROR] Unsupported file type")
                            st.stop()               
                    # 파일 업로드 후, 업로드된 파일 삭제
                    del uploaded_file
                    shutil.rmtree(upload_dir)
                
                if st.session_state['document_type'] == "Google Search":
                    st.session_state['document_source'] = google_search(st.session_state['google_search_query'], num_results=st.session_state['google_search_result_count'], lang=st.session_state['google_search_doc_lang'])
                    if st.session_state['google_custom_urls']:
                        custom_urls = st.session_state['google_custom_urls'].splitlines()
                        for url in custom_urls:
                            if is_valid_url(url):
                                st.session_state['document_source'].append(url)
                    if not st.session_state['document_source']:
                        st.error("[ERROR] No search results found.")
                        st.stop()
                    else:
                        st.session_state['document_source'] = list(dict.fromkeys(st.session_state['document_source']).keys())
                    for url in st.session_state['document_source']:
                        print(f"[DEBUG] (Loaded URL) {url}")
                        loader = WebBaseLoader(
                            web_paths = (url,),
                            # bs_kwargs=dict(
                            #     parse_only=bs4.SoupStrainer(
                            #         class_=("post-content", "post-title", "post-header")
                            #     )
                            # ),
                            show_progress = True,
                        )
                        docs_contents.append(loader.load())

                ### 문서 내용 정리 (빈 줄 제거)
                if st.session_state['document_type'] == "Google Search":
                    for i in range(len(docs_contents)):
                        if docs_contents[i]:
                            docs_contents[i][0].page_content = "\n".join([line for line in docs_contents[i][0].page_content.split("\n") if line.strip()])
                else:
                    docs_contents[0].page_content = "\n".join([line for line in docs_contents[0].page_content.split("\n") if line.strip()])

                # 문서 분할
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state['chunk_size'], chunk_overlap=st.session_state['chunk_overlap'])
                splits = []
                if st.session_state['document_type'] == "Google Search":
                    for doc_contents in docs_contents:
                        splits.extend(text_splitter.split_documents(doc_contents))
                else:
                    splits = text_splitter.split_documents(docs_contents)
                
                ### splits 의 각 split의 page_content에서 빈 줄 제거
                # splits = [Document(page_content="\n".join([line for line in split.page_content.split("\n") if line.strip()]), metadata=split.metadata) for split in splits]
                # 위 코드를 가독성을 위해 아래와 같이 수정
                # cleaned_splits = []
                # for split in splits:
                #     lines = split.page_content.split("\n")
                #     cleaned_lines = [line for line in lines if line.strip()]
                #     cleaned_content = "\n".join(cleaned_lines)
                #     cleaned_split = Document(page_content=cleaned_content, metadata=split.metadata)
                #     cleaned_splits.append(cleaned_split)
                # splits = cleaned_splits

                ### 디버깅용 출력
                # for idx, split in enumerate(splits):
                #     print(f"[DEBUG] Split {idx + 1}: {split.page_content[:200]}...")  # Print first 200 characters of each split

                # 모든 청크 벡터화 및 메타데이터 준비 (for Pinecone에 임베딩)
                if st.session_state['document_type'] == "Google Search":
                    for i, doc in enumerate(splits):
                        document = Document(
                            page_content=doc.page_content,
                            metadata={"id": i + 1, "source": st.session_state['document_source'][i // len(splits) * len(st.session_state['document_source'])]}
                        )
                        st.session_state.get('documents_chunks', []).append(document)
                else:
                    for i, doc in enumerate(splits):
                        document = Document(
                            page_content=doc.page_content,
                            metadata={"id": i + 1, "source": st.session_state['document_source'][0]}
                        )
                        st.session_state.get('documents_chunks', []).append(document)

                ### Debugging Print
                print(f"[DEBUG] (chunks) {len(st.session_state.get('documents_chunks', []))}")
                
                st.write(f"Documents Chunks: {len(st.session_state.get('documents_chunks', []))}")

                if st.session_state['vectorstore_type'] == "Pinecone":
                    # Pinecone 관련 설정
                    pc = Pinecone()
                    
                    ### Pinecone Index Name 설정 : 단순/동일한 이름 사용
                    pinecone_index_name = 'perplexis-' + st.session_state['selected_ai']
                    pinecone_index_name = pinecone_index_name.lower()
                    
                    ### Pinecone Index Name 설정 : Google Search Query 또는 URL 정보를 이용
                    # if st.session_state['document_type'] == "Google Search":
                    #     pc_index_name_tag = re.sub(r'[^a-zA-Z0-9]', '-', st.session_state.get("google_search_query", 'Unknown'))
                    # else:
                    #     pc_index_name_tag = re.sub(r'[^a-zA-Z0-9]', '-', st.session_state["document_source"][0])
                    # pinecone_index_name = 'perplexis--' + pc_index_name_tag
                    # pinecone_index_name = pinecone_index_name[:40] + "--end"
                    # pinecone_index_name = pinecone_index_name.lower()
                    
                    print(f"[DEBUG] (pinecone_index_name) {pinecone_index_name}")

                    # Pinecone Index 초기화 (삭제)
                    if st.session_state.get('pinecone_index_reset', False):
                        if pinecone_index_name in pc.list_indexes().names():
                            pc.delete_index(pinecone_index_name)

                    # 지정된 이름의 Pinecone Index가 존재하지 않으면, 신규 생성
                    if (pinecone_index_name not in pc.list_indexes().names()):
                        pc.create_index(
                            name = pinecone_index_name,
                            dimension = st.session_state.get('selected_embedding_dimension', None),
                            metric = st.session_state['pinecone_similarity'],
                            spec = ServerlessSpec(
                                cloud='aws',
                                region='us-east-1'
                            )
                        )

                    # Pinecone Embedding 처리
                    vectorstore = PineconeVectorStore.from_documents(
                        documents = st.session_state.get('documents_chunks', []),
                        embedding = st.session_state['embedding_instance'],
                        index_name = pinecone_index_name
                    )
                    
                if st.session_state['vectorstore_type'] == "ChromaDB":
                    chromadb_dir_path = f"{chromadb_root}/Perplexis-{st.session_state.get('selected_ai', 'unknown')}"
                    
                    if st.session_state.get('chromadb_root_reset', True):
                        reset_chromadb(chromadb_dir_path)
                    
                    vectorstore = Chroma.from_documents(
                        documents = st.session_state.get('documents_chunks', []),
                        embedding = st.session_state['embedding_instance'],
                        ### collection_metadata 문서 --> https://docs.trychroma.com/guides#changing-the-distance-function
                        # collection_metadata={"hnsw:space": "l2"},
                        # collection_metadata={"hnsw:space": "ip"},
                        # collection_metadata={"hnsw:space": "cosine"},
                        collection_metadata={"hnsw:space": st.session_state['chromadb_similarity']},
                        persist_directory = f"{chromadb_dir_path}",
                    )
                    
                if st.session_state['vectorstore_type'] == "PGVector":
                    # connection = f"postgresql+psycopg://{os.environ['PGVECTOR_USER']}:{os.environ['PGVECTOR_PASS']}@{os.environ['PGVECTOR_HOST']}:{os.environ['PGVECTOR_PORT']}/perplexis"
                    
                    if st.session_state.get('pgvector_db_reset', True):
                        reset_pgvector(st.session_state['pgvector_connection'])
                    
                    vectorstore = PGVector(
                        connection = st.session_state['pgvector_connection'],
                        embeddings = st.session_state['embedding_instance'],
                        embedding_length = st.session_state.get('selected_embedding_dimension', None),
                        collection_name = 'perplexis_rag',
                        ### "distance_strategy" : Should be one of l2, cosine, inner.
                        # distance_strategy = "l2",
                        # distance_strategy = "inner",
                        # distance_strategy = "cosine",
                        distance_strategy = st.session_state['pgvector_similarity'],
                        use_jsonb=True,
                    )
                    
                    vectorstore.add_documents(st.session_state.get('documents_chunks', []), ids=[doc.metadata["id"] for doc in st.session_state.get('documents_chunks', [])])

                # 주어진 문서 내용 처리(임베딩)
                if st.session_state['rag_search_type'] == "similarity_score_threshold":
                    search_kwargs = {"k": int(st.session_state['rag_top_k']), "score_threshold": float(st.session_state['rag_score_threshold'])}
                elif st.session_state['rag_search_type'] == "similarity":
                    search_kwargs = {"k": int(st.session_state['rag_top_k'])}
                elif st.session_state['rag_search_type'] == "mmr":
                    search_kwargs = {"k": int(st.session_state['rag_top_k']), "fetch_k": int(st.session_state['rag_fetch_k']), "lambda_mult": float(st.session_state['rag_lambda_mult'])}

                st.session_state['retriever'] = vectorstore.as_retriever(
                    search_type=st.session_state['rag_search_type'],
                    search_kwargs=search_kwargs
                )
                
                if st.session_state['retriever']:
                    st.success("Embedding completed!")
                else:
                    st.error("[ERROR] Embedding failed!")
                    st.stop()

                # RAG Chain 생성
                history_aware_retriever = create_history_aware_retriever(
                    st.session_state['llm'],
                    st.session_state['retriever'],
                    contextualize_q_prompt
                )

                question_answer_chain = create_stuff_documents_chain(st.session_state['llm'], qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                st.session_state['store'] = {}
                def get_session_history(session_id: str) -> BaseChatMessageHistory:
                    if session_id not in st.session_state['store']:
                        st.session_state['store'][session_id] = ChatMessageHistory()
                    return st.session_state['store'][session_id]

                st.session_state['conversational_rag_chain'] = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )

            st.session_state['is_analyzed'] = True            
            st.rerun() # UI 및 session_state 갱신

        if st.session_state.get('is_analyzed', False) == True:
            if st.button("Reset", type='primary'):
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
                # st.experimental_set_query_params(page="chat_and_rag")
                    
#-----------------------------------------------------------------------------------------------------------

    ### Google Search 결과 표시
    if st.session_state['document_type'] == "Google Search":
        # for idx, url in enumerate(st.session_state['document_source'], 1):
        #     st.write(f"URL-{idx}\t{url}")
        # multiline_text = "\n".join(st.session_state['document_source'])
        # st.info(multiline_text.replace("\n", "<br>"), icon="ℹ️", unsafe_allow_html=True)
        multiline_text = "".join(
            [f'<a href="{url}" target="_blank">{url}</a><br>' for url in st.session_state['document_source']]
        )
        st.markdown(
            f"""
            <div style="border: 1px solid #ddd; padding: 10px; background-color: #e7f3fc; 
                        font-family: Arial, sans-serif; border-radius: 5px; width: 100%;">
                <p><b>ℹ️ Google Search Results</b></p>
                <p style="white-space: pre-line;">{multiline_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

#-----------------------------------------------------------------------

    # 메인 창 로딩 가능 여부(retriever 객체 존재) 확인
    try:
        if not (st.session_state['retriever'] and st.session_state['session_id']) or not st.session_state['is_analyzed']:
            st.stop()
    except Exception as e:
        print(f"[Exception]: {e}")
        st.error("Please enter the required information in the left sidebar.")
        st.stop()

    ## Container 선언 순서가 화면에 보여지는 순서 결정
    container_history = st.container()
    container_user_textbox = st.container()

    ### container_user_textbox 처리
    with container_user_textbox:
        with st.form(key='user_textbox', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100, placeholder="Type your message here...")
            submit_button = st.form_submit_button(label='Send', use_container_width=False, help="Click to send your message", on_click=None, args=None, kwargs=None, disabled=False, icon=":material/send:", type='primary')

        if submit_button and user_input:
            with st.spinner('Thinking...'):
                result = st.session_state['conversational_rag_chain'].invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state['session_id']}
                    },
                )

            ### Debugging Print
            print(f"[DEBUG] AI Result: {result}")
            
            ai_response = result['answer']
            
            rag_contexts = result.get('context', [])
            
            ### Debugging Print
            # print("<RAG 데이터>")
            # for context in rag_contexts:
            #     print(context)

            llm_model_name = get_llm_model_name()

            st.session_state['rag_history_user'].append(user_input)
            st.session_state['rag_history_ai'].append(ai_response)
            st.session_state['rag_history_llm_model_name'].append(llm_model_name)
            st.session_state['rag_history_rag_contexts'].append(rag_contexts)
            st.session_state['rag_history_temperature'].append(st.session_state['temperature'])
            st.session_state['rag_history_rag_search_type'].append(st.session_state.get('rag_search_type', 'Unknown'))
            st.session_state['rag_history_rag_top_k'].append(st.session_state.get('rag_top_k', 'Unknown'))
            if st.session_state.get('rag_search_type', 'Unknown') == "similarity_score_threshold":
                curr_rag_score_threshold = st.session_state.get('rag_score_threshold', 'Unknown')
            else:
                curr_rag_score_threshold = "Unknown"
            st.session_state['rag_history_rag_score_threshold'].append(curr_rag_score_threshold)
            st.session_state['rag_history_rag_fetch_k'].append(st.session_state.get('rag_fetch_k', 'Unknown'))
            st.session_state['rag_history_rag_rag_lambda_mult'].append(st.session_state.get('rag_lambda_mult', 'Unknown'))

    ### container_history 처리
    if st.session_state['rag_history_ai']:
        with container_history:
            for i in range(len(st.session_state['rag_history_ai'])):
                # message(st.session_state["rag_history_user"][i], is_user=True, key=str(i) + '_user')
                # message(st.session_state["rag_history_ai"][i], key=str(i))
                
                with st.chat_message("user"):
                    st.markdown("**<span style='color: blue;'>You</span>**", unsafe_allow_html=True)
                    st.write(st.session_state["rag_history_user"][i])
                with st.chat_message("assistant"):
                    st.markdown(f"**<span style='color: green;'>{st.session_state.get('ai_role', 'Unknown')}</span>**", unsafe_allow_html=True)
                    st.write(st.session_state["rag_history_ai"][i])
                
                llm_model_name = st.session_state['rag_history_llm_model_name'][i]
                temperature = st.session_state['rag_history_temperature'][i]
                rag_search_type = st.session_state['rag_history_rag_search_type'][i]
                rag_top_k = st.session_state['rag_history_rag_top_k'][i]
                rag_fetch_k = st.session_state['rag_history_rag_fetch_k'][i]
                rag_lambda_mult = st.session_state['rag_history_rag_rag_lambda_mult'][i]
                if st.session_state.get('rag_search_type', 'Unknown') == "similarity_score_threshold":
                    rag_score_threshold = st.session_state['rag_history_rag_score_threshold'][i]
                else:
                    rag_score_threshold = None
                
                #st.write(f"Embeddings: {st.session_state.get('selected_embedding_provider', 'Unknown Embeddings')} / AI: {st.session_state.get('selected_ai', 'Unknown AI')} / LLM: {llm_model_name} / Temperature: {temperature} / RAG Contexts: {len(st.session_state['rag_history_rag_contexts'][-1])} / Pinecone Similarity: {st.session_state.get('pinecone_similarity', 'Unknown')}")
                #st.write(f"RAG Top-K: {rag_top_k} / RAG Search Type: {rag_search_type} / RAG Score: {st.session_state.get('rag_score_threshold', 'Unknown')} / RAG Fetch-K: {rag_fetch_k} / RAG Lambda Mult: {rag_lambda_mult}")

                # st.markdown(f"""
                #     <p style='color: #2E9AFE;'>
                #         - <b>Embeddings</b>: {st.session_state.get('selected_embedding_provider', 'Unknown Embeddings')}<br>
                #         - <b>AI</b>: {st.session_state.get('selected_ai', 'Unknown AI')}<br>
                #         - <b>LLM</b>: {llm_model_name}<br>
                #         - <b>Temperature</b>: {temperature}<br>
                #         - <b>RAG Contexts</b>: {len(st.session_state['rag_history_rag_contexts'][-1])}<br>
                #         - <b>Pinecone Similarity</b>: {st.session_state.get('pinecone_similarity', 'Unknown')}<br>
                #         - <b>RAG Top-K</b>: {rag_top_k}<br>
                #         - <b>RAG Search Type</b>: {rag_search_type}<br>
                #         - <b>RAG Score</b>: {st.session_state.get('rag_score_threshold', 'Unknown')}<br>
                #         - <b>RAG Fetch-K</b>: {rag_fetch_k}<br>
                #         - <b>RAG Lambda Mult</b>: {rag_lambda_mult}<br>
                #     </p>
                # """, unsafe_allow_html=True)
                
                with st.expander("Show Metadata"):
                    st.markdown(
                        f"""
                        <div style="color: #2E9AFE; border: 1px solid #ddd; padding: 5px; background-color: #f9f9f9; border-radius: 5px; width: 100%;">
                            - <b>Embedding Type</b>: <font color=black>{st.session_state.get('selected_embedding_provider', 'Unknown Embedding Type')}</font><br>
                            - <b>Embedding Model</b>: <font color=black>{st.session_state.get('selected_embedding_model', 'Unknown Embedding Model')}</font><br>
                            - <b>Embedding Dimension</b>: <font color=black>{st.session_state.get('selected_embedding_dimension', 'Unknown Embedding Dimension')}</font><br>
                            - <b>Chunk-Count</b>: <font color=black>{len(st.session_state.get('documents_chunks', []))}</font><br>
                            - <b>Chunk-Size</b>: <font color=black>{st.session_state.get('chunk_size', None)}</font><br>
                            - <b>Chunk-Overlap</b>: <font color=black>{st.session_state.get('chunk_overlap', None)}</font><br>
                            - <b>VectorDB Type</b>: <font color=black>{st.session_state.get('vectorstore_type', None)}</font><br>
                            - <b>AI</b>: <font color=red>{st.session_state.get('selected_ai', 'Unknown AI')}</font><br>
                            - <b>LLM Model</b>: <font color=black>{llm_model_name}</font><br>
                            - <b>LLM Args(Common) temperature</b>: <font color=black>{temperature}</font><br>
                            - <b>LLM Args(Common) top_p</b>: <font color=black>{st.session_state.get('llm_top_p', None)}</font><br>
                            - <b>LLM Args(OpenAI) presence_penalty</b>: <font color=black>{st.session_state.get('llm_openai_presence_penalty', None)}</font><br>
                            - <b>LLM Args(OpenAI) frequency_penalty</b>: <font color=black>{st.session_state.get('llm_openai_frequency_penalty', None)}</font><br>
                            - <b>LLM Args(OpenAI) max_tokens</b>: <font color=black>{st.session_state.get('llm_openai_max_tokens', None)}</font><br>
                            - <b>LLM Args(Ollama) repeat_penalty</b>: <font color=black>{st.session_state.get('llm_ollama_repeat_penalty', None)}</font><br>
                            - <b>LLM Args(Ollama) num_ctx</b>: <font color=black>{st.session_state.get('llm_ollama_fnum_ctx', None)}</font><br>
                            - <b>LLM Args(Ollama) num_predict</b>: <font color=black>{st.session_state.get('llm_ollama_num_predict', None)}</font><br>
                            - <b>Pinecone Similarity</b>: <font color=black>{st.session_state.get('pinecone_similarity', None)}</font><br>
                            - <b>RAG Contexts</b>: <font color=black>{len(st.session_state['rag_history_rag_contexts'][-1])}</font><br>
                            - <b>RAG Top-K</b>: <font color=black>{rag_top_k}</font><br>
                            - <b>RAG Search Type</b>: <font color=black>{rag_search_type}</font><br>
                            - <b>RAG Score</b>: <font color=black>{rag_score_threshold}</font><br>
                            - <b>RAG Fetch-K</b>: <font color=black>{rag_fetch_k}</font><br>
                            - <b>RAG Lambda Mult</b>: <font color=black>{rag_lambda_mult}</font><br>
                            
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write("")
                
                # print(f"[DEBUG] (rag_history_rag_contexts) {st.session_state['rag_history_rag_contexts']}")
                
                # 소스 데이터 표시
                with st.expander("Show Contexts"):
                    if st.session_state['rag_history_rag_contexts'][i]:
                        for context in st.session_state['rag_history_rag_contexts'][i]:
                            st.write(context)
                    else:
                        st.write("No Contexts")

#-----------------------------------------------------------------------

if __name__ == "__main__":
    Navbar()
    
    main()
    # # Error 메세지/코드 숨기기
    # try:
    #     main()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
