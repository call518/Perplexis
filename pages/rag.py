__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

### pip requirements generate
# pip list --format=freeze > requirements.txt

import os
import re
import shutil
import time
import uuid
import requests
import json
import fitz
import bs4
import asyncio
import chromadb
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from googlesearch import search as GoogleSearch
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

from langchain.chains import create_history_aware_retriever, create_retrieval_chain, ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, PyMuPDFLoader, TextLoader
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
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Chroma
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

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

from langchain.callbacks.base import BaseCallbackHandler  # 추가

#--------------------------------------------------

# 페이지 정보 정의
st.set_page_config(page_title="Perplexis:Rag", page_icon=":books:", layout="wide")
st.title(":material/quick_reference_all: _:red[Perplexis]_ RAG")

### Mandatory Keys 설정
### 1. secretes.toml 파일에 설정된 KEY 값이 최우선 적용.
### 2. secretes.toml 파일에 설정된 KEY 값이 없을 경우, os.environ 환경변수로 설정된 KEY 값이 적용.
### 3. secretes.toml 파일과 os.environ 환경변수 모두 설정되지 않은 경우, Default 값을 적용.

def set_env_var_if_not_exists(secret_key_name, env_var_name, default_val=""):
    """Checks environment variables or secrets for specific keys and sets defaults if missing."""
    if st.secrets["KEYS"].get(secret_key_name):
        os.environ[env_var_name] = st.secrets["KEYS"].get(secret_key_name)
    elif not os.environ.get(env_var_name):
        os.environ[env_var_name] = default_val

# Use the helper function to set environment variables for OLLAMA
set_env_var_if_not_exists("OLLAMA_BASE_URL", "OLLAMA_BASE_URL", "http://localhost:11434")

# Use the helper function to set environment variables for OPENAI
set_env_var_if_not_exists("OPENAI_BASE_URL", "OPENAI_BASE_URL", "https://api.openai.com/v1")
set_env_var_if_not_exists("OPENAI_API_KEY", "OPENAI_API_KEY")

# Use the helper function to set environment variables for PINECONE
set_env_var_if_not_exists("PINECONE_API_KEY", "PINECONE_API_KEY")

# Use the helper function to set environment variables for PGVector
set_env_var_if_not_exists("PGVECTOR_HOST", "PGVECTOR_HOST", "localhost")
set_env_var_if_not_exists("PGVECTOR_PORT", "PGVECTOR_PORT", "5432")
set_env_var_if_not_exists("PGVECTOR_USER", "PGVECTOR_USER", "perplexis")
set_env_var_if_not_exists("PGVECTOR_PASS", "PGVECTOR_PASS", "changeme")

# google search api key & custom search engine id
set_env_var_if_not_exists("GOOGLE_API_KEY", "GOOGLE_API_KEY")
set_env_var_if_not_exists("GOOGLE_CSE_ID", "GOOGLE_CSE_ID")

# telegram api key & chat id
set_env_var_if_not_exists("TELEGRAM_TOKEN", "TELEGRAM_TOKEN")
set_env_var_if_not_exists("TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID")

# langchain api key
if st.secrets["KEYS"].get("LANGCHAIN_API_KEY", None):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["KEYS"].get("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = "Perplexis"

def get_remote_ip() -> str:
    """Get remote ip."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            st.error(f"[ERROR] Client IP 확인 오류 발생: ctx is None")
            st.stop()

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            st.error(f"[ERROR] Client IP 확인 오류 발생: session_info is None")
            st.stop()
    except Exception as e:
        st.error(f"[ERROR] Client IP 확인 오류 발생: {e}")
        st.stop()

    return session_info.request.remote_ip

default_values = {
    'ai_role': None,
    'history_aware_retriever': None,
    'chain_with_history': None,
    'question_answer_chain': None,
    'rag_chain': None,
    'session_id': str(uuid.uuid4()),
    'client_remote_ip' : get_remote_ip(),
    'is_analyzed': False,
    'document_type': None,
    'document_source': [],
    'documents_chunks': [],
    'embedding_instance': None,
    'llm': None,
    'llm_top_p': 0.50,
    'llm_openai_presence_penalty': 0.00,
    'llm_openai_frequency_penalty': 1.00,
    'llm_openai_max_tokens': 1024,
    'llm_ollama_repeat_penalty': 1.00,
    'llm_ollama_num_ctx': None,
    'llm_ollama_num_predict': None,
    'selected_embedding_provider': None,
    'selected_embedding_model': None,
    'selected_embedding_dimension': None,
    'vectorstore_type': None,
    'vectorstore_instance': None,
    'pgvector_connection': None,
    'pinecone_similarity': None,
    'chromadb_similarity': None,
    'pgvector_similarity': None,
    'selected_ai': None,
    'selected_llm': None,
    'temperature': 0.05,
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
    'rag_history': [],  # single list for storing Q/A with metadata
    'store': {},
}

def init_session_state():
    """Initializes session state with default values and cleans up any existing uploads."""
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
        if key not in ['store', 'documents_chunks', 'rag_history']:
            print(f"[DEBUG] (session_state) {key} = {st.session_state[key]}")
    os.system('rm -rf ./uploads/*')

init_session_state()
upload_dir = f"./uploads/{st.session_state.session_id}"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

url_pattern = re.compile(
    r'^(?:http|ftp)s?://'  # http:// 또는 https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # 도메인 이름
    r'localhost|'  # localhost
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IPv4 주소
    r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # IPv6 주소
    r'(?::\d+)?'  # 포트 번호
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def google_search(query, num_results=10, lang="English(en)"):
    results_list = []
    lang = re.search(r"\((.*?)\)", lang).group(1)
    try:
        ### (최우선) Google Custom Search API 우선 사용
        if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"):
            ### (Document) https://developers.google.com/custom-search/docs/xml_results?hl=ko
            url = "https://www.googleapis.com/customsearch/v1"
            response = requests.get(url, params={
                "key" : os.environ.get('GOOGLE_API_KEY'),
                "cx" : os.environ.get('GOOGLE_CSE_ID'),
                "q" : query,
                "num": 10, # Maxixum 10
                "lr": f'lang_{lang}',
                "hl": lang,
            })
            content = json.loads(response.content)
            if content:
                for idx, item in enumerate(content['items'], 1):
                    link = item['link']
                    print(f"[DEBUG] (Google API Search URLs) {idx}. {link}")
                    results_list.append(link)
                return results_list
            else:
                st.error("No search results found.")
                st.stop()
        ### (차선) Google Custom Search API 미사용 시, Google Search 사용
        else:
            results = GoogleSearch(query, num_results=num_results, lang=lang)
            if results:
                for idx, link in enumerate(results, 1):
                    print(f"[DEBUG] (Google Search URLs) {idx}. {link}")
                    results_list.append(link)
                return results_list
            else:
                st.error("No search results found.")
                st.stop()
    except Exception as e:
        st.error(f"[ERROR] {e}")
        st.stop()

def get_llm_model_name():
    """Return the name of the LLM model."""
    ai_model = st.session_state.llm
    if ai_model:
        llm_model_name = ai_model.model_name if hasattr(ai_model, 'model_name') else ai_model.model
    else:
        llm_model_name = "Unknown LLM"
    return llm_model_name

def reset_pgvector(connection):
    """Drops and recreates PGVector tables to ensure a clean starting point."""
    engine = create_engine(connection)
    Session = sessionmaker(bind=engine)
    session = Session()
    tables = ["langchain_pg_embedding", "langchain_pg_collection"]
    for table in tables:
        session.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    session.commit()
    session.close()

chromadb_root = './chromadb'
def reset_chromadb(db_path=chromadb_root):
    """Removes all persisted ChromaDB data for a new indexing process."""
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path)

def is_valid_url(url):
    """Check if the URL is valid."""
    return re.match(url_pattern, url) is not None

def read_uri_content(uri, type):
    """Read content from a given URI and return as Document."""
    if type == "FILE-TXT":
        loader = TextLoader(uri)
        docs = loader.load()
    elif type == "FILE-PDF":
        loader = PyMuPDFLoader(uri, extract_images=True)
        docs = loader.load()
    elif type == "URL-PDF":
        res = requests.get(uri)
        contents = fitz.open(stream=res.content, filetype="pdf")
        docs = []
        for page_num in range(contents.page_count):
            page_text = contents.get_page_text(pno=page_num)
            docs.append([Document(page_content=page_text, metadata={"source": uri, "page": page_num + 1})])
    else:
        st.error("[ERROR] read_uri_content()")
        st.stop()
    return docs

def create_embedding_instance(provider):
    if provider == "OpenAI":
        st.session_state.embedding_instance = OpenAIEmbeddings(
            base_url = os.environ.get("OPENAI_BASE_URL"),
            model = st.session_state.selected_embedding_model,
        )
    else:
        st.session_state.embedding_instance = OllamaEmbeddings(
            base_url = os.environ.get("OLLAMA_BASE_URL"),
            model = st.session_state.selected_embedding_model,
        )

# 추가: Streamlit 콜백 핸들러
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.placeholder.write(self.text)

#-----------------------------------------------------------

# asyncio.run(telegramSendMessage("Perplexis:Rag", os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")))

def main():
    """Governs the Streamlit UI for retrieving and embedding documents, then performing RAG."""
    with st.sidebar:
        st.title("Parameters")
        st.session_state.ai_role = st.selectbox("Role of AI", get_ai_role_and_sysetm_prompt(only_key=True), index=0)
        st.session_state.document_type = st.radio("Document Type", ("URL", "File Upload", "Google Search"), horizontal=True, disabled=st.session_state.is_analyzed)

        if st.session_state.document_type == "URL":
            st.session_state.document_source = [ st.text_input("Document URL", disabled=st.session_state.is_analyzed) ]
        elif st.session_state.document_type == "File Upload":
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                with open(f"{upload_dir}/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.write(f"File uploaded: {uploaded_file.name}")
                st.session_state.document_source.append(uploaded_file.name)
        elif st.session_state.document_type == "Google Search":
            st.session_state.google_search_query = st.text_input("Google Search Query", placeholder="Enter Your Keywords")
            st.session_state.google_custom_urls = st.text_area("Google Search Custom URLs (per Line)", placeholder="Enter Your Keywords")
            col_google_search_result_count, col_google_search_doc_lang = st.sidebar.columns(2)
            with col_google_search_result_count:
                if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"):
                    search_result_max_count = 10
                else:
                    search_result_max_count = 50
                st.session_state.google_search_result_count = st.number_input("Search Results", min_value=1, max_value=search_result_max_count, value=10, step=1)
            with col_google_search_doc_lang:
                st.session_state.google_search_doc_lang = st.selectbox("Search Document Language", ["Afrikaans(af)", "Albanian(sq)", "Amharic(am)", "Arabic(ar)", "Armenian(hy)", "Azerbaijani(az)", "Bengali(bn)", "Bulgarian(bg)", "Burmese(my)", "Catalan(ca)", "Chinese (Simplified)(zh-CN)", "Chinese (Traditional)(zh-TW)", "Croatian(hr)", "Czech(cs)", "Danish(da)", "Dutch(nl)", "English(en)", "English (UK)(en-GB)", "Estonian(et)", "Filipino(fil)", "Finnish(fi)", "French(fr)", "French (Canadian)(fr-CA)", "Georgian(ka)", "German(de)", "Greek(el)", "Gujarati(gu)", "Hebrew(iw)", "Hindi(hi)", "Hungarian(hu)", "Icelandic(is)", "Indonesian(id)", "Italian(it)", "Japanese(ja)", "Kannada(kn)", "Kazakh(kk)", "Khmer(km)", "Korean(ko)", "Kyrgyz(ky)", "Laothian(lo)", "Latvian(lv)", "Lithuanian(lt)", "Macedonian(mk)", "Malay(ms)", "Malayam(ml)", "Marathi(mr)", "Mongolian(mn)", "Nepali(ne)", "Norwegian (Bokmal)(no)", "Persian(fa)", "Polish(pl)", "Portuguese (Brazil)(pt-BR)", "Portuguese (Portugal)(pt-PT)", "Punjabi(pa)", "Romanian(ro)", "Russian(ru)", "Serbian(sr)", "Serbian (Latin)(sr-Latn)", "Sinhalese(si)", "Slovak(sk)", "Slovenian(sl)", "Spanish(es)", "Spanish (Latin America)(es-419)", "Swahili(sw)", "Swedish(sv)", "Tamil(ta)", "Telugu(te)", "Thai(th)", "Turkish(tr)", "Ukrainian(uk)", "Urdu(ur)", "Uzbek(uz)", "Vietnamese(vi)", "Welsh(cy)"], index=16)
        else:
            st.error("[ERROR] Unsupported document type")
            st.stop()

        col_ai, col_embedding, col_vectorstore = st.sidebar.columns(3)
        with col_ai:
            st.session_state.selected_ai = st.radio("**:blue[AI]**", ("Ollama", "OpenAI"), index=0)
        with col_embedding:
            st.session_state.selected_embedding_provider = st.radio("**:blue[Embeddings]**", ("Ollama", "OpenAI"), index=0, disabled=st.session_state.is_analyzed)
        with col_vectorstore:
            st.session_state.vectorstore_type = st.radio("**:blue[VectorDB]**", ("ChromaDB", "PGVector", "Pinecone"), index=0, disabled=st.session_state.is_analyzed)
        
        if st.session_state.selected_embedding_provider == "OpenAI" or st.session_state.selected_ai == "OpenAI":
            os.environ["OPENAI_API_KEY"] = st.text_input("**:red[OpenAI API Key]** [Learn more](https://platform.openai.com/docs/quickstart)", value=os.environ["OPENAI_API_KEY"], type="password")
            os.environ["OPENAI_BASE_URL"] = st.text_input("OpenAI API URL", value=os.environ.get("OPENAI_BASE_URL"))
    
        if st.session_state.selected_embedding_provider == "Ollama" or st.session_state.selected_ai == "Ollama":
            os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama API URL", value=os.environ.get("OLLAMA_BASE_URL"))
    
        if st.session_state.vectorstore_type == "ChromaDB":
            st.session_state.chromadb_root_reset = st.checkbox("Reset ChromaDB", value=st.session_state.chromadb_root_reset, disabled=st.session_state.is_analyzed)
            st.session_state.chromadb_similarity = st.selectbox("ChromaDB Similarity", ["cosine", "l2", "ip"], index=0)

        if st.session_state.vectorstore_type == "PGVector":
            st.session_state.pgvector_db_reset = st.checkbox("Reset PGVector", value=st.session_state.pgvector_db_reset, disabled=st.session_state.is_analyzed)

            if st.session_state.pgvector_connection is None:
                st.session_state.pgvector_connection = f"postgresql+psycopg://{os.environ.get('PGVECTOR_USER')}:{os.environ.get('PGVECTOR_PASS')}@{os.environ.get('PGVECTOR_HOST')}:{os.environ.get('PGVECTOR_PORT')}/perplexis"
            st.session_state.pgvector_connection = st.text_input("PGVector Connection", value=st.session_state.pgvector_connection, type="password",  disabled=st.session_state.is_analyzed)
            st.session_state.pgvector_similarity = st.selectbox("PGVector Similarity", ["cosine", "euclidean", "dotproduct"], index=0)
    
        if st.session_state.vectorstore_type == "Pinecone":
            os.environ["PINECONE_API_KEY"] = st.text_input("**:red[Pinecone API Key]** [Learn more](https://www.pinecone.io/docs/quickstart/)", value=os.environ.get("PINECONE_API_KEY"), type="password")
            st.session_state.pinecone_index_reset = st.checkbox("Reset Pinecone Index", value=st.session_state.pinecone_index_reset, disabled=st.session_state.is_analyzed)
            st.session_state.pinecone_similarity = st.selectbox("Pinecone Similarity", ["cosine", "l2", "inner"], index=0)

        if st.session_state.selected_embedding_provider == "OpenAI":
            st.session_state.selected_embedding_model = st.selectbox("Embedding Model", ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"], index=0,  disabled=st.session_state.is_analyzed)
        else:
            st.session_state.selected_embedding_model = st.selectbox("Embedding Model", ["bge-m3:567m", "all-minilm:22m", "all-minilm:33m", "nomic-embed-text", "mxbai-embed-large", "gemma2:2b", "gemma2:9b", "gemma2:27b", "call518/gemma2-uncensored-8192ctx:9b", "mistral:7b", "llama3:8b", "llama3.2:1b", "llama3.2:3b", "call518/deepseek-r1-32768ctx:8b", "call518/deepseek-r1-32768ctx:14b", "call518/EEVE-8192ctx:10.8b-q4", "call518/EEVE-8192ctx:10.8b-q5", "llama3-groq-tool-use:8b", "call518/tulu3-131072ctx:8b", "call518/exaone3.5-32768ctx:2.4b", "call518/exaone3.5-32768ctx:7.8b", "call518/exaone3.5-32768ctx:32b"], index=4, disabled=st.session_state.is_analyzed)

        st.session_state.selected_embedding_dimension = get_max_value_of_model_embedding_dimensions(st.session_state.selected_embedding_model)
        print(f"[DEBUG] (selected_embedding_model) {st.session_state.selected_embedding_model}")
        print(f"[DEBUG] (selected_embedding_dimension) {st.session_state.selected_embedding_dimension}")
        if st.session_state.selected_embedding_dimension is None:
            st.error("[ERROR] Unsupported embedding model")
            st.stop()

        col_ai_llm, col_ai_temperature = st.sidebar.columns(2)
        with col_ai_llm:
            if st.session_state.selected_ai == "OpenAI":
                st.session_state.selected_llm = st.selectbox("AI LLM", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], index=0)
            else:
                st.session_state.selected_llm = st.selectbox("AI LLM", ["gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b", "gemma2:2b", "gemma2:9b", "gemma2:27b", "call518/gemma2-uncensored-8192ctx:9b", "mistral:7b", "llama3:8b", "llama3.2:1b", "llama3.2:3b", "codegemma:2b", "codegemma:7b", "call518/deepseek-r1-32768ctx:8b", "call518/deepseek-r1-32768ctx:14b", "call518/EEVE-8192ctx:10.8b-q4", "call518/EEVE-8192ctx:10.8b-q5", "llama3-groq-tool-use:8b", "call518/tulu3-131072ctx:8b", "call518/exaone3.5-32768ctx:2.4b", "call518/exaone3.5-32768ctx:7.8b", "call518/exaone3.5-32768ctx:32b"], index=0)
        with col_ai_temperature:
            st.session_state.temperature = st.number_input("AI Temperature", min_value=0.00, max_value=1.00, value=st.session_state.temperature, step=0.05)

        col_chunk_size, col_chunk_overlap = st.sidebar.columns(2)
        with col_chunk_size:
            st.session_state.chunk_size = st.number_input("Chunk Size", min_value=200, max_value=5000, value=500, step=100)
        with col_chunk_overlap:
            st.session_state.chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=100, step=100)
            if not (st.session_state.chunk_size >= (st.session_state.chunk_overlap * 2)):
                st.error("Chunk Overlap must be less than Chunk Size.")
                st.stop()

        st.session_state.rag_search_type = st.selectbox("RAG Search Type", ["similarity", "similarity_score_threshold", "mmr"], index=1)
    
        if st.session_state.rag_search_type == "similarity_score_threshold":
            col_rag_arg1, col_rag_arg2 = st.sidebar.columns(2)
            with col_rag_arg1:
                st.session_state.rag_top_k = st.number_input("RAG Top-K", min_value=1, max_value=50, value=10, step=1)
            with col_rag_arg2:
                st.session_state.rag_score_threshold = st.number_input("score_threshold", min_value=0.01, max_value=1.00, value=0.60, step=0.05)
        elif st.session_state.rag_search_type == "similarity":
            st.session_state.rag_top_k = st.number_input("RAG Top-K", min_value=1, max_value=50, value=10, step=1)
        elif st.session_state.rag_search_type == "mmr":
            col_rag_arg1, col_rag_arg2, col_rag_arg3 = st.sidebar.columns(3)
            with col_rag_arg1:
                st.session_state.rag_top_k = st.number_input("RAG Top-K", min_value=1, max_value=50, value=10, step=1)
            with col_rag_arg2:
                st.session_state.rag_fetch_k = st.number_input("MMR Fetch-K", min_value=1, value=5, step=1)
            with col_rag_arg3:
                st.session_state.rag_lambda_mult = st.number_input("MMR Lambda Mult", min_value=0.01, max_value=1.00, value=0.50, step=0.05)

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

        if st.session_state.selected_embedding_provider == "OpenAI" or st.session_state.selected_ai == "OpenAI":
            if not os.environ.get("OPENAI_API_KEY"):
                st.error("Please enter the OpenAI API Key.")
                st.stop()

        if st.session_state.vectorstore_type == "Pinecone":
            if not os.environ.get("PINECONE_API_KEY"):
                st.error("Please enter the Pinecone API Key.")
                st.stop()

        if st.session_state.selected_embedding_provider == "OpenAI":
            st.session_state.embedding_instance = OpenAIEmbeddings(
                base_url = os.environ.get("OPENAI_BASE_URL"),
                model = st.session_state.selected_embedding_model,
            )
        else:
            st.session_state.embedding_instance = OllamaEmbeddings(
                base_url = os.environ.get("OLLAMA_BASE_URL"),
                model = st.session_state.selected_embedding_model,
            )
        # create_embedding_instance(st.session_state.selected_embedding_provider)

        if st.button("Embedding", disabled=st.session_state.is_analyzed):
            docs_contents = []

            with st.spinner('Embedding...'):
                if st.session_state.document_type == "URL":
                    if not st.session_state.document_source:
                        st.error("[ERROR] URL information is missing.")
                        st.stop()

                    if not is_valid_url(st.session_state.document_source[0]):
                        st.error("[ERROR] Invalid URL.")
                        st.stop()
                    
                    loader = WebBaseLoader(
                        web_paths = (st.session_state.document_source[0],),
                        show_progress = True,
                    )
                    docs_contents = loader.load()
                
                if st.session_state.document_type == "File Upload":
                    if uploaded_file is not None:
                        if uploaded_file.type == "application/pdf":
                            docs_contents = read_uri_content(f"{upload_dir}/{uploaded_file.name}", "FILE-PDF")
                        elif uploaded_file.type == "text/plain":
                            docs_contents = read_uri_content(f"{upload_dir}/{uploaded_file.name}", "FILE-TXT")
                        else:
                            st.error("[ERROR] Unsupported file type")
                            st.stop()               
                    del uploaded_file
                    shutil.rmtree(upload_dir)
                
                if st.session_state.document_type == "Google Search":
                    st.session_state.document_source = google_search(st.session_state.google_search_query, num_results=st.session_state.google_search_result_count, lang=st.session_state.google_search_doc_lang)
                    print(f"[DEBUG] (document_source) ----------------> {st.session_state.document_source}")
                    if st.session_state.google_custom_urls:
                        custom_urls = st.session_state.google_custom_urls.splitlines()
                        for url in custom_urls:
                            if is_valid_url(url):
                                st.session_state.document_source.append(url)
                    if not st.session_state.document_source:
                        st.error("[ERROR] No search results found.")
                        st.stop()
                    else:
                        st.session_state.document_source = list(dict.fromkeys(st.session_state.document_source).keys())

                    for url in st.session_state.document_source:
                        print(f"[DEBUG] (Loaded URL) {url}")
                        try:
                            url_head = requests.head(url, allow_redirects=True)
                        except requests.RequestException as e:
                            print(f"[ERROR] Failed to connect to URL {url}: {e}")
                            print(f"[INFO] Skipping URL {url} and continuing with the next one.")
                            continue
                        url_content_type = url_head.headers.get('Content-Type', '')
                        if 'application/pdf' in url_content_type:
                            doc_pdf_page_list = read_uri_content(url, "URL-PDF")
                            docs_contents.extend(doc_pdf_page_list)
                        else:
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

                if st.session_state.document_type == "Google Search":
                    for i in range(len(docs_contents)):
                        if docs_contents[i]:
                            docs_contents[i][0].page_content = "\n".join([line for line in docs_contents[i][0].page_content.split("\n") if line.strip()])
                else:
                    docs_contents[0].page_content = "\n".join([line for line in docs_contents[0].page_content.split("\n") if line.strip()])

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
                splits = []
                if st.session_state.document_type == "Google Search":
                    for doc_contents in docs_contents:
                        splits.extend(text_splitter.split_documents(doc_contents))
                else:
                    splits = text_splitter.split_documents(docs_contents)

                if st.session_state.document_type == "Google Search":
                    for i, doc in enumerate(splits):
                        document = Document(
                            page_content=doc.page_content,
                            metadata={"id": i + 1, "source": st.session_state.document_source[i // len(splits) * len(st.session_state.document_source)]}
                        )
                        st.session_state.documents_chunks.append(document)
                else:
                    for i, doc in enumerate(splits):
                        document = Document(
                            page_content=doc.page_content,
                            metadata={"id": i + 1, "source": st.session_state.document_source[0]}
                        )
                        st.session_state.documents_chunks.append(document)

                print(f"[DEBUG] (chunks) {len(st.session_state.documents_chunks)}")
                print(f"[DEBUG] (selected_embedding_dimension) {st.session_state.selected_embedding_dimension}")
                
                st.write(f"Documents Chunks: {len(st.session_state.documents_chunks)}")

                if st.session_state.vectorstore_type == "Pinecone":
                    pc = Pinecone()
                    pinecone_index_name = 'perplexis-' + st.session_state.selected_ai
                    pinecone_index_name = pinecone_index_name.lower()
                    print(f"[DEBUG] (pinecone_index_name) {pinecone_index_name}")

                    if st.session_state.pinecone_index_reset:
                        if pinecone_index_name in pc.list_indexes().names():
                            pc.delete_index(pinecone_index_name)

                    if (pinecone_index_name not in pc.list_indexes().names()):
                        pc.create_index(
                            name = pinecone_index_name,
                            dimension = st.session_state.selected_embedding_dimension,
                            metric = st.session_state.pinecone_similarity,
                            spec = ServerlessSpec(
                                cloud='aws',
                                region='us-east-1'
                            )
                        )

                    ### Embedding 데이터 추가
                    st.session_state.vectorstore_instance = PineconeVectorStore.from_documents(
                        documents = st.session_state.documents_chunks,
                        embedding = st.session_state.embedding_instance,
                        index_name = pinecone_index_name
                    )
                    
                if st.session_state.vectorstore_type == "ChromaDB":
                    chromadb_dir_path = f"{chromadb_root}/Perplexis-{st.session_state.selected_ai}"
                    
                    if st.session_state.chromadb_root_reset:
                        reset_chromadb(chromadb_dir_path)
                    
                    ### Embedding 데이터 추가
                    st.session_state.vectorstore_instance = Chroma.from_documents(
                        documents = st.session_state.documents_chunks,
                        embedding = st.session_state.embedding_instance,
                        collection_metadata={"hnsw:space": st.session_state.chromadb_similarity},
                        persist_directory = f"{chromadb_dir_path}",
                    )
                    
                if st.session_state.vectorstore_type == "PGVector":
                    if st.session_state.pgvector_db_reset:
                        reset_pgvector(st.session_state.pgvector_connection)
                    
                    st.session_state.vectorstore_instance = PGVector(
                        connection = st.session_state.pgvector_connection,
                        embeddings = st.session_state.embedding_instance,
                        embedding_length = st.session_state.selected_embedding_dimension,
                        collection_name = 'perplexis_rag',
                        ### "distance_strategy" : Should be one of l2, cosine, inner.
                        # distance_strategy = "l2",
                        # distance_strategy = "inner",
                        # distance_strategy = "cosine",
                        distance_strategy = st.session_state.pgvector_similarity,
                        use_jsonb=True,
                    )
                    
                    ### Embedding 데이터 추가
                    st.session_state.vectorstore_instance.add_documents(st.session_state.documents_chunks, ids=[doc.metadata["id"] for doc in st.session_state.documents_chunks])

                if st.session_state.vectorstore_instance:
                    st.success("Embedding completed!")
                else:
                    st.error("[ERROR] Embedding failed!")
                    st.stop()

            st.session_state.is_analyzed = True
            st.rerun()

        if st.session_state.is_analyzed == True:
            if st.button("Reset", type='primary'):
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
                    
    if st.session_state.document_type == "Google Search":
        multiline_text = "".join(
            [f'<a href="{url}" target="_blank">{url}</a><br>' for url in st.session_state.document_source]
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

    try:
        if not st.session_state.session_id or not st.session_state.is_analyzed:
            st.stop()
    except Exception as e:
        print(f"[Exception]: {e}")
        st.error("Please enter the required information in the left sidebar.")
        st.stop()

    ### RAG Main Windows
    container_history = st.container()
    container_user_textbox = st.container()

    with container_user_textbox:
        with st.form(key='user_textbox', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100, placeholder="Type your message here...")
            submit_button = st.form_submit_button(label='Send', use_container_width=False, help="Click to send your message", on_click=None, args=None, kwargs=None, disabled=False, icon=":material/send:", type='primary')

        if submit_button and user_input:
            # streaming_placeholder = st.empty()
            # callback_handler = StreamlitCallbackHandler(placeholder=streaming_placeholder)

            contextualize_q_system_prompt = (
                # "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
                "Given a chat history and the latest user question, formulate a standalone question that can be understood without the chat history while still preserving key contextual elements needed for retrieval. Ensure that relevant information from prior exchanges is retained to maintain continuity for effective retrieval. Do NOT answer the question; just reformulate it if necessary. If the latest user question depends on prior context, incorporate the necessary details from chat history concisely while avoiding excessive verbosity."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            system_prompt_ai_role = get_ai_role_and_sysetm_prompt(st.session_state.ai_role)
            system_prompt = (
                "\n"
                "- If there is no content between <CTX-START> and <CTX-END>, or if the context lacks sufficient information, under no circumstances should you provide an answer. (Write a message stating that you cannot answer because there is no context to refer to, and then end the task.)\n"
                "- If there is content between <CTX-START> and <CTX-END>, but some of it is unrelated to the user's question, that content should be ignored."
                "- If you don't know the answer, say you don't know.\n"
                "- Write the answer using at least 50 sentences if possible, and include as much accurate and detailed information as possible.\n"
                "- Write the answer in Korean if possible.\n"
                f"- Your role as an AI is as follows: {system_prompt_ai_role}\n"
                "- After answering the question, provide five relevant follow-up questions under the heading 'Suggested Questions'. These questions should be closely related to the content of your answer, encouraging deeper exploration or clarification of key points.\n"

                "\n"
                "<CTX-START>\n"
                "{context}\n"
                "<CTX-END>\n"
            )
            # print(f"[DEBUG] (system_prompt) {system_prompt}")
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            ### Retriver 제정의
            if st.session_state.rag_search_type == "similarity_score_threshold":
                search_kwargs = {
                    "k": int(st.session_state.rag_top_k),
                    "score_threshold": float(st.session_state.rag_score_threshold)
                }
            elif st.session_state.rag_search_type == "similarity":
                search_kwargs = {
                    "k": int(st.session_state.rag_top_k)
                }
            elif st.session_state.rag_search_type == "mmr":
                search_kwargs = {
                    "k": int(st.session_state.rag_top_k),
                    "fetch_k": int(st.session_state.rag_fetch_k),
                    "lambda_mult": float(st.session_state.rag_lambda_mult)
                }
            st.session_state.retriever = st.session_state.vectorstore_instance.as_retriever(
                search_type=st.session_state.rag_search_type,
                search_kwargs=search_kwargs,
            )

            ### LLM 제정의
            if st.session_state.selected_ai == "OpenAI":
                st.session_state.llm = ChatOpenAI(
                    base_url = os.environ.get("OPENAI_BASE_URL"),
                    model = st.session_state.selected_llm,
                    temperature = st.session_state.temperature,
                    cache = False,
                    # streaming = True,
                    streaming = False,
                    # callbacks = [callback_handler],
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
                    base_url = os.environ.get("OLLAMA_BASE_URL"),
                    model = st.session_state.selected_llm,
                    temperature = st.session_state.temperature,
                    cache = False,
                    # streaming = True,
                    streaming = False,
                    # callbacks = [callback_handler],
                    num_ctx = st.session_state.llm_ollama_num_ctx,
                    num_predict = st.session_state.llm_ollama_num_predict,
                    # num_gpu = None,
                    # num_thread = None,
                    # repeat_last_n = None,
                    repeat_penalty = st.session_state.llm_ollama_repeat_penalty,
                    # tfs_z = None,
                    # top_k = None,
                    top_p = st.session_state.llm_top_p,
                    # format = "", # Literal['', 'json'] (default: "")
                    verbose = True,
                )

            # if st.session_state.history_aware_retriever is None:
            st.session_state.history_aware_retriever = create_history_aware_retriever(
                st.session_state.llm,
                st.session_state.retriever,
                contextualize_q_prompt
            )

            st.session_state.question_answer_chain = create_stuff_documents_chain(st.session_state.llm, qa_prompt)
            st.session_state.rag_chain = create_retrieval_chain(st.session_state.history_aware_retriever, st.session_state.question_answer_chain)

            # if not st.session_state.store:
            #     st.session_state.store = {}

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            st.session_state.chain_with_history = RunnableWithMessageHistory(
                st.session_state.rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            with st.spinner('Thinking...'):
                start_time = time.perf_counter()
                result = st.session_state.chain_with_history.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                )
                end_time = time.perf_counter()
                answer_elapsed_time = end_time - start_time

            print(f"[DEBUG] (answer_elapsed_time) {answer_elapsed_time} sec")
            # print(f"[DEBUG] AI Result: {result}")
            
            ai_response = result['answer']
            rag_contexts = result.get('context', [])
            new_entry = {
                "user": user_input,
                "assistant": ai_response,
                "role_of_ai": st.session_state.ai_role,
                "llm_model_name": get_llm_model_name(),
                "rag_contexts": rag_contexts,
                "temperature": st.session_state.temperature,
                "rag_search_type": st.session_state.rag_search_type,
                "rag_top_k": st.session_state.rag_top_k,
                "rag_score_threshold": st.session_state.rag_score_threshold,
                "rag_fetch_k": st.session_state.rag_fetch_k,
                "rag_lambda_mult": st.session_state.rag_lambda_mult,
                "elapsed_time": answer_elapsed_time,
            }
            st.session_state.rag_history.append(new_entry)
            
            # Telegram Send Message
            new_entry_str = "\n".join([f"{key}: {value}" for key, value in new_entry.items() if key not in ('user', 'assistant', 'rag_contexts')])
            document_source_str = "\n".join(st.session_state.document_source)
            full_message = "===== Perplexis:Rag =====\n\n[User]\n" + user_input + "\n\n[Assistant]\n" + ai_response + "\n\n[Metadata]\n" + new_entry_str + "\n\n[Document Sources]\n" + document_source_str
            try:
                asyncio.run(telegramSendMessage(full_message, os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")))
            except Exception as e:
                print(f"[ERROR] Telegram messaging failed: {e}")

            # streaming_placeholder.empty()  # 추가: 스트리밍 출력 후 placeholder 비우기

    if st.session_state.rag_history:
        with container_history:
            for i, entry in enumerate(st.session_state.rag_history):
                with st.chat_message("user"):
                    st.markdown("**<span style='color: blue;'>You</span>**", unsafe_allow_html=True)
                    st.write(entry["user"])
                with st.chat_message("assistant"):
                    st.markdown(f"**<span style='color: green;'>{st.session_state.ai_role}</span>**", unsafe_allow_html=True)
                    st.write(entry["assistant"])
                
                llm_model_name = entry["llm_model_name"]
                temperature = entry["temperature"]
                rag_search_type = entry["rag_search_type"]
                rag_top_k = entry["rag_top_k"]
                rag_fetch_k = entry["rag_fetch_k"]
                rag_lambda_mult = entry["rag_lambda_mult"]
                if st.session_state.rag_search_type == "similarity_score_threshold":
                    rag_score_threshold = entry["rag_score_threshold"]
                else:
                    rag_score_threshold = None
                answer_elapsed_time = entry["elapsed_time"]
                
                with st.expander("Show Metadata"):
                    st.markdown(
                        f"""
                        <div style="color: #2E9AFE; border: 1px solid #ddd; padding: 5px; background-color: #f9f9f9; border-radius: 5px; width: 100%;">
                            - <b>Embedding Type</b>: <font color=black>{st.session_state.selected_embedding_provider}</font><br>
                            - <b>Embedding Model</b>: <font color=black>{st.session_state.selected_embedding_model}</font><br>
                            - <b>Embedding Dimension</b>: <font color=black>{st.session_state.selected_embedding_dimension}</font><br>
                            - <b>Chunk-Count</b>: <font color=black>{len(st.session_state.documents_chunks)}</font><br>
                            - <b>Chunk-Size</b>: <font color=black>{st.session_state.chunk_size}</font><br>
                            - <b>Chunk-Overlap</b>: <font color=black>{st.session_state.chunk_overlap}</font><br>
                            - <b>VectorDB Type</b>: <font color=black>{st.session_state.vectorstore_type}</font><br>
                            - <b>AI</b>: <font color=red>{st.session_state.selected_ai}</font><br>
                            - <b>Role of AI</b>: <font color=black>{st.session_state.ai_role}</font><br>
                            - <b>LLM Model</b>: <font color=black>{llm_model_name}</font><br>
                            - <b>LLM Args(Common) temperature</b>: <font color=black>{temperature}</font><br>
                            - <b>LLM Args(Common) top_p</b>: <font color=black>{st.session_state.llm_top_p}</font><br>
                            - <b>LLM Args(OpenAI) presence_penalty</b>: <font color=black>{st.session_state.llm_openai_presence_penalty}</font><br>
                            - <b>LLM Args(OpenAI) frequency_penalty</b>: <font color=black>{st.session_state.llm_openai_frequency_penalty}</font><br>
                            - <b>LLM Args(OpenAI) max_tokens</b>: <font color=black>{st.session_state.llm_openai_max_tokens}</font><br>
                            - <b>LLM Args(Ollama) repeat_penalty</b>: <font color=black>{st.session_state.llm_ollama_repeat_penalty}</font><br>
                            - <b>LLM Args(Ollama) num_ctx</b>: <font color=black>{st.session_state.llm_ollama_num_ctx}</font><br>
                            - <b>LLM Args(Ollama) num_predict</b>: <font color=black>{st.session_state.llm_ollama_num_predict}</font><br>
                            - <b>Pinecone Similarity</b>: <font color=black>{st.session_state.pinecone_similarity}</font><br>
                            - <b>RAG Contexts</b>: <font color=black>{len(entry["rag_contexts"])}</font><br>
                            - <b>RAG Top-K</b>: <font color=black>{rag_top_k}</font><br>
                            - <b>RAG Search Type</b>: <font color=black>{rag_search_type}</font><br>
                            - <b>RAG Score</b>: <font color=black>{rag_score_threshold}</font><br>
                            - <b>RAG Fetch-K</b>: <font color=black>{rag_fetch_k}</font><br>
                            - <b>RAG Lambda Mult</b>: <font color=black>{rag_lambda_mult}</font><br>
                            - <b>Elapsed time</b>: <font color=black>{answer_elapsed_time:.0f} sec</font><br>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write("")
                
                with st.expander("Show Contexts"):
                    if entry["rag_contexts"]:
                        for context in entry["rag_contexts"]:
                            st.write(context)
                    else:
                        st.write(f"No relevant docs were retrieved using the relevance score threshold {rag_score_threshold}")

if __name__ == "__main__":
    Navbar()
    
    main()
