# pip install -U langchain-community bs4 langchain_pinecone pinecone-client[grpc] langchain-openai streamlit-chat streamlit-js-eval pypdf googlesearch-python chromadb pysqlite3-binary
# pip freeze > requirements.txt
# pip list --format=freeze > requirements.txt

### (임시) pysqlite3 설정 - sqlite3 모듈을 pysqlite3로 대체
### pip install pysqlite3-binary 필요
### "Your system has an unsupported version of sql requires sqlite3 >= 3.35.0." 오류 해결 목적
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from modules.nav import Navbar
from modules.common_functions import get_sysetm_prompt_for_role
from modules.common_functions import get_country_name_by_code

import streamlit as st
# from streamlit_chat import message
from streamlit_js_eval import streamlit_js_eval
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
from langchain_community.document_loaders import TextLoader
from googlesearch import search

import os
import shutil
import requests

# 페이지 정보 정의
st.set_page_config(page_title="Perplexis:Rag", page_icon=":books:", layout="wide")
st.title(":books: _:red[Perplexis]_ Chat & RAG")

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
    'selected_mode': "Chat",
    'ai_role': None,
    'system_prompt_content': None,
    'chat_memory': None,
    'chat_conversation': None,
    'chat_response': None,
    'session_id': str(uuid.uuid4()),
    'client_remote_ip' : get_remote_ip(),
    'is_analyzed': False,
    'vectorstore_type': "pinecone",
    'vectorstore_dimension': None,
    'document_type': None,
    'document_source': [],
    'documents_chunks': [],
    'embeddings': None,
    'llm': None,
    'selected_embedding': None,
    'selected_ai': None,
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
        print(f"[DEBUG] (session_state) {key} = {st.session_state.get(key, '')}")
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
                if not result.lower().endswith(".pdf"):
                    try:
                        response = requests.head(result, allow_redirects=True)
                        if 'application/pdf' not in response.headers.get('Content-Type', ''):
                            results_list.append(result)
                            #print(f"[DEBUG] (Google Search URLs) {idx}. {result}")
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
    # upload_dir = f"./uploads/{st.session_state['session_id']}"
    loader = PyPDFLoader(f"{upload_dir}/{file}")
    docs = loader.load()
    # shutil.rmtree(upload_dir)
    return docs

#--------------------------------------------------

def main():
    # 사이드바 영역 구성
    with st.sidebar:
        st.title("Parameters")

        st.session_state['selected_mode'] = st.radio("**:red[Mode]**", ("Chat", "RAG"), horizontal=True, disabled=st.session_state['is_analyzed'])

        st.session_state['ai_role'] = st.selectbox("Role of AI", get_sysetm_prompt_for_role(only_key=True), index=0)

        if st.session_state.get('selected_mode', "Chat") == "RAG":
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
            system_prompt_content = get_sysetm_prompt_for_role(st.session_state['ai_role'])
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

        if st.session_state.get('selected_mode', "Chat") == "RAG":
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
                    st.session_state['google_search_result_count'] = st.number_input("Search Results", min_value=1, max_value=50, value=5, step=1, disabled=st.session_state['is_analyzed'])
                with col_google_search_doc_lang:
                    st.session_state['google_search_doc_lang'] = st.selectbox("Search Document Language", [ "Any", "en", "ko", "jp", "cn"], disabled=st.session_state['is_analyzed'])
            else:
                st.error("[ERROR] Unsupported document type")
                st.stop()

        if st.session_state.get('selected_mode', "Chat") == "RAG":
            col_ai, col_embedding, col_vectorstore = st.sidebar.columns(3)
            with col_ai:
                st.session_state['selected_ai'] = st.radio("**:blue[AI]**", ("Ollama", "OpenAI"), index=0, disabled=st.session_state['is_analyzed'])
            with col_embedding:
                st.session_state['selected_embeddings'] = st.radio("**:blue[Embeddings]**", ("Ollama", "OpenAI"), index=0, disabled=st.session_state['is_analyzed'])
            with col_vectorstore:
                st.session_state['vectorstore_type'] = st.radio("**:blue[VectorDB]**", ("ChromaDB", "Pinecone"), index=0, disabled=st.session_state['is_analyzed'])
        else:
            st.session_state['selected_ai'] = st.radio("**:blue[AI]**", ("Ollama", "OpenAI"), index=0, disabled=st.session_state['is_analyzed'])
        

        if st.session_state.get('selected_mode', "Chat") == "RAG":
            if st.session_state.get('selected_embeddings', "Ollama") == "OpenAI" or st.session_state.get('selected_ai', "Ollama") == "OpenAI":
                os.environ["OPENAI_API_KEY"] = st.text_input("**:red[OpenAI API Key]** [Learn more](https://platform.openai.com/docs/quickstart)", value=os.environ["OPENAI_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
                os.environ["OPENAI_BASE_URL"] = st.text_input("OpenAI API URL", value=os.environ["OPENAI_BASE_URL"], disabled=st.session_state['is_analyzed'])
        
            if st.session_state.get('selected_embeddings', "Ollama") == "Ollama" or st.session_state.get('selected_ai', "Ollama") == "Ollama":
                os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama API URL", value=os.environ["OLLAMA_BASE_URL"], disabled=st.session_state['is_analyzed'])
        
            if st.session_state.get('vectorstore_type', 'ChromaDB') == "ChromaDB":
                st.session_state['chromadb_root_reset'] = st.checkbox("Reset ChromaDB", value=st.session_state.get('chromadb_root_reset', True), disabled=st.session_state['is_analyzed'])
        
            if st.session_state.get('vectorstore_type', 'ChromaDB') == "Pinecone":
                os.environ["PINECONE_API_KEY"] = st.text_input("**:red[Pinecone API Key]** [Learn more](https://www.pinecone.io/docs/quickstart/)", value=os.environ["PINECONE_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
                st.session_state['pinecone_index_reset'] = st.checkbox("Reset Pinecone Index", value=st.session_state.get('pinecone_index_reset', False), disabled=st.session_state['is_analyzed'])
                st.session_state['pinecone_metric'] = st.selectbox("Pinecone Metric", ["cosine", "euclidean", "dotproduct"], disabled=st.session_state['is_analyzed'])
        else:
            if st.session_state.get('selected_ai', "Ollama") == "OpenAI":
                os.environ["OPENAI_API_KEY"] = st.text_input("**:red[OpenAI API Key]** [Learn more](https://platform.openai.com/docs/quickstart)", value=os.environ["OPENAI_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
                os.environ["OPENAI_BASE_URL"] = st.text_input("OpenAI API URL", value=os.environ["OPENAI_BASE_URL"], disabled=st.session_state['is_analyzed'])

        col_ai_llm, col_ai_temperature = st.sidebar.columns(2)
        with col_ai_llm:
            if st.session_state['selected_ai'] == "OpenAI":
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], index=4,  disabled=st.session_state['is_analyzed'])
            else:
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gemma2", "mistral", "llama3.2", "codegemma"], disabled=st.session_state['is_analyzed'])
        with col_ai_temperature:
            # st.session_state['temperature'] = st.text_input("LLM Temperature (0.0 ~ 1.0)", value=st.session_state['temperature'])
            st.session_state['temperature'] = st.number_input("AI Temperature", min_value=0.00, max_value=1.00, value=st.session_state['temperature'], step=0.05, disabled=st.session_state['is_analyzed'])
        
        if st.session_state.get('selected_mode', "Chat") == "RAG":
            col_chunk_size, col_chunk_overlap = st.sidebar.columns(2)
            with col_chunk_size:
                st.session_state['chunk_size'] = st.number_input("Chunk Size", min_value=500, max_value=5000, value=1000, step=100, disabled=st.session_state['is_analyzed'])
            with col_chunk_overlap:
                st.session_state['chunk_overlap'] = st.number_input("Chunk Overlap", min_value=100, max_value=1000, value=200, step=100, disabled=st.session_state['is_analyzed'])
                if st.session_state['chunk_size'] <= st.session_state['chunk_overlap']:
                    st.error("Chunk Overlap must be less than Chunk Size.")
                    st.stop()

            ### (임시) similarity_score_threshold 모드에서, score_threshold 값 사용 안되는 버그 있음. (해결될 때 까지 노출 안함...)
            #st.session_state['rag_search_type'] = st.selectbox("RAG Search Type", ["similarity", "similarity_score_threshold", "mmr"], index=0, disabled=st.session_state['is_analyzed'])
            st.session_state['rag_search_type'] = st.selectbox("RAG Search Type", ["similarity", "mmr"], index=0, disabled=st.session_state['is_analyzed'])
        
            if st.session_state['rag_search_type'] == "similarity_score_threshold":
                col_rag_arg1, col_rag_arg2 = st.sidebar.columns(2)
                with col_rag_arg1:
                    st.session_state['rag_top_k'] = st.number_input("TOP-K", min_value=1, value=20, step=1, disabled=st.session_state['is_analyzed'])        
                with col_rag_arg2:
                    st.session_state['rag_score'] = st.number_input("Score", min_value=0.01, max_value=1.00, value=0.60, step=0.05, disabled=st.session_state['is_analyzed'])
            elif st.session_state['rag_search_type'] == "similarity":
                st.session_state['rag_top_k'] = st.number_input("TOP-K", min_value=1, value=20, step=1, disabled=st.session_state['is_analyzed'])
            elif st.session_state['rag_search_type'] == "mmr":
                col_rag_arg1, col_rag_arg2, col_rag_arg3 = st.sidebar.columns(3)
                with col_rag_arg1:
                    st.session_state['rag_top_k'] = st.number_input("TOP-K", min_value=1, value=20, step=1, disabled=st.session_state['is_analyzed'])
                with col_rag_arg2:
                    st.session_state['rag_fetch_k'] = st.number_input("Fetch-K", min_value=1, value=5, step=1, disabled=st.session_state['is_analyzed'])
                with col_rag_arg3:
                    st.session_state['rag_lambda_mult'] = st.number_input("Lambda Mult", min_value=0.01, max_value=1.00, value=0.80, step=0.05, disabled=st.session_state['is_analyzed'])

        if st.session_state.get('selected_mode', "Chat") == "RAG":
            if st.session_state.get('selected_embeddings', "Ollama") == "OpenAI" or st.session_state.get('selected_ai', "Ollama") == "OpenAI":
                if not os.environ["OPENAI_API_KEY"]:
                    st.error("Please enter the OpenAI API Key.")
                    st.stop()
        elif st.session_state.get('selected_mode', "Chat") == "Chat":
            if st.session_state.get('selected_ai', "Ollama") == "OpenAI":
                if not os.environ["OPENAI_API_KEY"]:
                    st.error("Please enter the OpenAI API Key.")
                    st.stop()

        if st.session_state.get('selected_mode', "Chat") == "RAG":
            if st.session_state.get('vectorstore_type', 'Pinecone') == "Pinecone":
                if not os.environ["PINECONE_API_KEY"]:
                    st.error("Please enter the Pinecone API Key.")
                    st.stop()

        if st.session_state.get('selected_mode', "Chat") == "RAG":
            # Embedding 선택 및 초기화
            if st.session_state['selected_embeddings'] == "OpenAI":
                st.session_state['embeddings'] = OpenAIEmbeddings(
                    api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_BASE_URL"],
                    model="text-embedding-3-large" # dimension = 3072
                )
                st.session_state['vectorstore_dimension'] = 3072
            else:
                st.session_state['embeddings'] = OllamaEmbeddings(
                    base_url=os.environ["OLLAMA_BASE_URL"],
                    model="mxbai-embed-large",  # dimension = 1024
                )
                st.session_state['vectorstore_dimension'] = 1024

        # AI 모델 선택 및 초기화
        if st.session_state['selected_ai'] == "OpenAI":
            st.session_state['llm'] = ChatOpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_BASE_URL"],
                model=st.session_state['selected_llm'],
                temperature=st.session_state['temperature'],
            )
        else:
            st.session_state['llm'] = Ollama(
                base_url=os.environ["OLLAMA_BASE_URL"],
                model=st.session_state['selected_llm'],
                temperature=st.session_state['temperature'],
            )

        if st.session_state.get('selected_mode', "Chat") == "RAG":
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
                            web_paths=(st.session_state['document_source'][0],),
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
                                web_paths=(url,),
                            )
                            docs_contents.append(loader.load())

                    # 문서 분할
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state['chunk_size'], chunk_overlap=st.session_state['chunk_overlap'])
                    splits = []
                    if st.session_state['document_type'] == "Google Search":
                        for doc_contents in docs_contents:
                            splits.extend(text_splitter.split_documents(doc_contents))
                    else:
                        splits = text_splitter.split_documents(docs_contents)

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
                    print(f"[DEBUG] Chunks ---------------> {len(st.session_state.get('documents_chunks', []))}")
                    
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
                        
                        print(f"[DEBUG] pinecone_index_name ---------------> {pinecone_index_name}")

                        # Pinecone Index 초기화 (삭제)
                        if st.session_state.get('pinecone_index_reset', False):
                            if pinecone_index_name in pc.list_indexes().names():
                                pc.delete_index(pinecone_index_name)

                        # 지정된 이름의 Pinecone Index가 존재하지 않으면, 신규 생성
                        if (pinecone_index_name not in pc.list_indexes().names()):
                            pc.create_index(
                                name=pinecone_index_name,
                                dimension=st.session_state['vectorstore_dimension'],
                                metric=st.session_state['pinecone_metric'],
                                spec=ServerlessSpec(
                                    cloud='aws',
                                    region='us-east-1'
                                )
                            )

                        # Pinecone Embedding 처리
                        vectorstore = PineconeVectorStore.from_documents(
                            st.session_state.get('documents_chunks', []),
                            st.session_state['embeddings'],
                            index_name=pinecone_index_name
                        )
                    else:
                        chromadb_dir_path = f"{chromadb_root}/Perplexis-{st.session_state.get('selected_ai', 'unknown')}"
                        
                        if st.session_state.get('chromadb_root_reset', True):
                            reset_chromadb(chromadb_dir_path)
                        
                        vectorstore = Chroma.from_documents(
                            st.session_state.get('documents_chunks', []),
                            st.session_state['embeddings'],
                            persist_directory=f"{chromadb_dir_path}",
                        )
                    
                    # 주어진 문서 내용 처리(임베딩)
                    if st.session_state['rag_search_type'] == "similarity_score_threshold":
                        search_kwargs = {"k": int(st.session_state['rag_top_k']), "score_threshold": st.session_state['rag_score']}
                    elif st.session_state['rag_search_type'] == "similarity":
                        search_kwargs = {"k": int(st.session_state['rag_top_k'])}
                    elif st.session_state['rag_search_type'] == "mmr":
                        search_kwargs = {"k": int(st.session_state['rag_top_k']), "fetch_k": int(st.session_state['rag_fetch_k']), "lambda_mult": st.session_state['rag_lambda_mult']}
                    st.session_state['retriever'] = vectorstore.as_retriever(search_type=st.session_state['rag_search_type'], search_kwargs=search_kwargs)
                    
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
                    
        if st.session_state.get('selected_mode', "Chat") == "Chat":
            if st.button("Reset", type='primary'):
                streamlit_js_eval(js_expressions="parent.window.location.reload()")

#-----------------------------------------------------------------------------------------------------------

    if st.session_state.get('selected_mode', "Chat") == "RAG":
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
                    <p><b>ℹ️ Google Search Results (Chunks: {len(st.session_state.get('documents_chunks', []))})</b></p>
                    <p style="white-space: pre-line;">{multiline_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

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
            with st.form(key='my_form', clear_on_submit=True):
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
                print(f"AI Result: {result}")
                
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
                    curr_rag_score = st.session_state.get('rag_score', 'Unknown')
                else:
                    curr_rag_score = "Unknown"
                st.session_state['rag_history_rag_score'].append(curr_rag_score)
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
                        rag_score = st.session_state['rag_history_rag_score'][i]
                    else:
                        rag_score = None
                    
                    #st.write(f"Embeddings: {st.session_state.get('selected_embeddings', 'Unknown Embeddings')} / AI: {st.session_state.get('selected_ai', 'Unknown AI')} / LLM: {llm_model_name} / Temperature: {temperature} / RAG Contexts: {len(st.session_state['rag_history_rag_contexts'][-1])} / Pinecone Metric: {st.session_state.get('pinecone_metric', 'Unknown')}")
                    #st.write(f"RAG TOP-K: {rag_top_k} / RAG Search Type: {rag_search_type} / RAG Score: {st.session_state.get('rag_score', 'Unknown')} / RAG Fetch-K: {rag_fetch_k} / RAG Lambda Mult: {rag_lambda_mult}")

                    # st.markdown(f"""
                    #     <p style='color: #2E9AFE;'>
                    #         - <b>Embeddings</b>: {st.session_state.get('selected_embeddings', 'Unknown Embeddings')}<br>
                    #         - <b>AI</b>: {st.session_state.get('selected_ai', 'Unknown AI')}<br>
                    #         - <b>LLM</b>: {llm_model_name}<br>
                    #         - <b>Temperature</b>: {temperature}<br>
                    #         - <b>RAG Contexts</b>: {len(st.session_state['rag_history_rag_contexts'][-1])}<br>
                    #         - <b>Pinecone Metric</b>: {st.session_state.get('pinecone_metric', 'Unknown')}<br>
                    #         - <b>RAG TOP-K</b>: {rag_top_k}<br>
                    #         - <b>RAG Search Type</b>: {rag_search_type}<br>
                    #         - <b>RAG Score</b>: {st.session_state.get('rag_score', 'Unknown')}<br>
                    #         - <b>RAG Fetch-K</b>: {rag_fetch_k}<br>
                    #         - <b>RAG Lambda Mult</b>: {rag_lambda_mult}<br>
                    #     </p>
                    # """, unsafe_allow_html=True)
                    
                    with st.expander("Show Metadata"):
                        st.markdown(
                            f"""
                            <div style="color: #2E9AFE; border: 1px solid #ddd; padding: 5px; background-color: #f9f9f9; border-radius: 5px; width: 100%;">
                                - <b>AI</b>: {st.session_state.get('selected_ai', 'Unknown AI')}<br>
                                - <b>LLM</b>: {llm_model_name}<br>
                                - <b>Embeddings</b>: {st.session_state.get('selected_embeddings', 'Unknown Embeddings')}<br>
                                - <b>Temperature</b>: {temperature}<br>
                                - <b>RAG Contexts</b>: {len(st.session_state['rag_history_rag_contexts'][-1])}<br>
                                - <b>Pinecone Metric</b>: {st.session_state.get('pinecone_metric', None)}<br>
                                - <b>RAG TOP-K</b>: {rag_top_k}<br>
                                - <b>RAG Search Type</b>: {rag_search_type}<br>
                                - <b>RAG Score</b>: {rag_score}<br>
                                - <b>RAG Fetch-K</b>: {rag_fetch_k}<br>
                                - <b>RAG Lambda Mult</b>: {rag_lambda_mult}<br>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.write("")
                    
                    # 소스 데이터 표시
                    if st.session_state["rag_history_rag_contexts"][i]:
                        with st.expander("Show Contexts"):
                            for context in st.session_state["rag_history_rag_contexts"][i]:
                                st.write(context)

#-----------------------------------------------------------------------

    ### Chat 모드 처리
    if st.session_state.get('selected_mode', "Chat") == "Chat":
        # system_prompt_content = "You are a chatbot having a conversation with a human."
        # system_prompt_content = "I want you to act as an academician. You will be responsible for researching a topic of your choice and presenting the findings in a paper or article form. Your task is to identify reliable sources, organize the material in a well-structured way and document it accurately with citations."
        st.session_state['system_prompt_content'] = get_sysetm_prompt_for_role(st.session_state.get('ai_role', "Basic chatbot"))
        
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
                    print(f"[DEBUG] chat_response ------------> {st.session_state['chat_response']}")

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
