# pip install -U langchain-community bs4 langchain_pinecone pinecone-client[grpc] langchain-openai streamlit-chat streamlit-js-eval pypdf googlesearch-python
# pip list --format=freeze > requirements.txt

import streamlit as st
from streamlit_chat import message
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
import re
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from googlesearch import search

import os
import shutil

### Mandatory Keys 설정
os.environ["OLLAMA_BASE_URL"] = st.secrets["KEYS"].get("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ["OPENAI_BASE_URL"] = st.secrets["KEYS"].get("OPENAI_BASE_URL", "https://api.openai.com/v1")
os.environ["OPENAI_API_KEY"] = st.secrets["KEYS"].get("OPENAI_API_KEY", "")
os.environ["PINECONE_API_KEY"] = st.secrets["KEYS"].get("PINECONE_API_KEY", "")

### (Optional) Langchain API Key 설정
if st.secrets["KEYS"].get("LANGCHAIN_API_KEY", ""):
    os.environ["LANGCHAIN_TRACING_V2"]= "true"
    os.environ["LANGCHAIN_ENDPOINT"]= "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["KEYS"].get("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"]= "Perplexis"

# 페이지 정보 정의
st.set_page_config(page_title="Perplexis", page_icon=":books:", layout="wide")
st.title(":books: _:red[[Perplexis](https://github.com/call518/Perplexis)]_")

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
    # 'session_id': str(uuid.uuid4()),
    'session_id': get_remote_ip(),
    'is_analyzed': False,
    'vectorstore_dimension': None,
    'document_type': None,
    'document_source': [],
    'embeddings': None,
    'llm': None,
    'selected_embedding': None,
    'selected_ai': None,
    'temperature': 0.00,
    'chunk_size': None,
    'chunk_overlap': None,
    'retriever': None,
    'rag_search_type': None,
    'rag_score': None,
    'rag_top_k': None,
    'google_search_result_count': None,
    'google_search_result_lang': None,
    'google_search_query': None,
    'chat_history_user': [],
    'chat_history_ai': [],
    'chat_history_llm_model_name': [],
    'chat_history_rag_contexts': [],
    'chat_history_temperature': [],
    'store': {},
}

def init_session_state():
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
        print(f"[session_state] {key} = {st.session_state.get(key, '')}")
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

def google_search(query, num_results=10, lang="ko"):
    results_list = []
    try:
        results = search(query, num_results=num_results, lang=lang)
        
        if results:
            for idx, result in enumerate(results, 1):
                results_list.append(result)
                # print(f"{idx}. {result}")
            return results_list
        else:
            print("검색 결과를 찾을 수 없습니다.")
            return []
    except Exception as e:
        st.error(f"[ERROR] 오류 발생: {e}")
        st.stop()

def get_llm_model_name():
    # AI 모델 정보 표시
    ai_model = st.session_state.get('llm', None)
    if ai_model:
        llm_model_name = ai_model.model_name if hasattr(ai_model, 'model_name') else ai_model.model
    else:
        llm_model_name = "Unknown LLM"
    return llm_model_name

def is_valid_url(url):
    return re.match(url_pattern, url) is not None

# Pinecone 관련 설정
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# pinecone_index_name = 'perplexis'

def read_txt(file):
    # upload_dir = f"./uploads/{st.session_state['session_id']}"
    loader = TextLoader(f"{upload_dir}/{file}")
    docs = loader.load()
    # shutil.rmtree(upload_dir)
    return docs

def read_pdf(file):
    # upload_dir = f"./uploads/{st.session_state['session_id']}"
    loader = PyPDFLoader(f"{upload_dir}/{file}")
    docs = loader.load()
    # shutil.rmtree(upload_dir)
    return docs

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
system_prompt = (
    # "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."
    # "당신은 모든 분야에서 전문가 입니다. 다음 검색된 컨텍스트 조각을 사용하여 질문자의 질문에 체계적인 답변을 작성하세요. 답을 모르면 모른다고 말하세요. 최소 50개의 문장을 사용하고 답변은 간결하면서도 최대한 상세한 정보를 포함해야 합니다. (답변은 한국어로 작성하세요.)"
    "You are an expert in all fields. Using the following retrieved context snippets, formulate a systematic answer to the asker's question. If you don't know the answer, say you don't know. Use at least 50 sentences, and ensure the response is concise yet contains as much detailed information as possible. (The response should be written in Korean.)"
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

#--------------------------------------------------

def main():
    # 사이드바 영역 구성
    with st.sidebar:
        st.title("Parameters")

        col_embedding, col_ai = st.sidebar.columns(2)
        with col_embedding:
            st.session_state['selected_embeddings'] = st.radio("Embedding", ("Ollama", "OpenAI"), index=1, disabled=st.session_state['is_analyzed'])
        with col_ai:
            st.session_state['selected_ai'] = st.radio("AI", ("Ollama", "OpenAI"), index=1, disabled=st.session_state['is_analyzed'])

        os.environ["OPENAI_API_KEY"] = st.text_input("**:red[OpenAI API Key]** [Learn more](https://platform.openai.com/docs/quickstart)", value=os.environ["OPENAI_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
        os.environ["OPENAI_BASE_URL"] = st.text_input("OpenAI API URL", value=os.environ["OPENAI_BASE_URL"], disabled=st.session_state['is_analyzed'])
        os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama API URL", value=os.environ["OLLAMA_BASE_URL"], disabled=st.session_state['is_analyzed'])
        os.environ["PINECONE_API_KEY"] = st.text_input("**:red[Pinecone API Key]** [Learn more](https://www.pinecone.io/docs/quickstart/)", value=os.environ["PINECONE_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])

        col_ai_llm, col_ai_temperature = st.sidebar.columns(2)
        with col_ai_llm:
            if st.session_state['selected_ai'] == "OpenAI":
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], disabled=st.session_state['is_analyzed'])
            else:
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gemma2", "mistral", "llama3.2", "codegemma"], disabled=st.session_state['is_analyzed'])
        with col_ai_temperature:
            # st.session_state['temperature'] = st.text_input("LLM Temperature (0.0 ~ 1.0)", value=st.session_state['temperature'])
            st.session_state['temperature'] = st.number_input("AI Temperature", min_value=0.00, max_value=1.00, value=st.session_state['temperature'], step=0.05, disabled=st.session_state['is_analyzed'])
        
        col_chunk_size, col_chunk_overlap = st.sidebar.columns(2)
        with col_chunk_size:
            st.session_state['chunk_size'] = st.number_input("Chunk Size", min_value=500, max_value=5000, value=1000, step=100, disabled=st.session_state['is_analyzed'])
        with col_chunk_overlap:
            st.session_state['chunk_overlap'] = st.number_input("Chunk Overlap", min_value=100, max_value=1000, value=200, step=100, disabled=st.session_state['is_analyzed'])
            if st.session_state['chunk_size'] <= st.session_state['chunk_overlap']:
                st.error("Chunk Overlap must be less than Chunk Size.")
                st.stop()

        col_pinecone_metric, col_rag_search_type = st.sidebar.columns(2)
        with col_pinecone_metric:
            st.session_state['pinecone_metric'] = st.selectbox("Pinecone Metric", ["cosine", "euclidean", "dotproduct"], disabled=st.session_state['is_analyzed'])
        with col_rag_search_type:
            st.session_state['rag_search_type'] = st.selectbox("RAG Search Type", ["similarity_score_threshold", "similarity", "mmr"], disabled=st.session_state['is_analyzed'])
        
        if st.session_state['rag_search_type'] == "similarity_score_threshold":
            col_rag_arg1, col_rag_arg2 = st.sidebar.columns(2)
            with col_rag_arg1:
                st.session_state['rag_top_k'] = st.number_input("TOP-K", min_value=1, value=5, step=1, disabled=st.session_state['is_analyzed'])        
            with col_rag_arg2:
                st.session_state['rag_score'] = st.number_input("Score", min_value=0.01, max_value=1.00, value=0.60, step=0.05, disabled=st.session_state['is_analyzed'])
        elif st.session_state['rag_search_type'] == "similarity":
            st.session_state['rag_top_k'] = st.number_input("TOP-K", min_value=1, value=5, step=1, disabled=st.session_state['is_analyzed'])
        elif st.session_state['rag_search_type'] == "mmr":
            col_rag_arg1, col_rag_arg2, col_rag_arg3 = st.sidebar.columns(3)
            with col_rag_arg1:
                st.session_state['rag_top_k'] = st.number_input("TOP-K", min_value=1, value=5, step=1, disabled=st.session_state['is_analyzed'])
            with col_rag_arg2:
                st.session_state['fetch_k'] = st.number_input("Fetch-K", min_value=1, value=5, step=1, disabled=st.session_state['is_analyzed'])
            with col_rag_arg3:
                st.session_state['lambda_mult'] = st.number_input("Lambda Mult", min_value=0.01, max_value=1.00, value=0.80, step=0.05, disabled=st.session_state['is_analyzed'])

        if not os.environ["OPENAI_API_KEY"]:
            st.error("Please enter the OpenAI API Key.")
            st.stop()

        if not os.environ["PINECONE_API_KEY"]:
            st.error("Please enter the Pinecone API Key.")
            st.stop()

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
            col_google_search_result_count, col_google_search_result_lang = st.sidebar.columns(2)
            with col_google_search_result_count:
                st.session_state['google_search_result_count'] = st.number_input("Search Results", min_value=1, max_value=50, value=5, step=1, disabled=st.session_state['is_analyzed'])
            with col_google_search_result_lang:
                st.session_state['google_search_result_lang'] = st.selectbox("Search Language", [ "en", "ko"], disabled=st.session_state['is_analyzed']) 
            
        else:
            st.error("[ERROR] Unsupported document type")
            st.stop()

        # 사용자 선택 및 입력값을 기본으로 RAG 데이터 준비
        if st.button("Embedding", disabled=st.session_state['is_analyzed']):
            docs_contents = []

            with st.spinner('Embedding...'):
                if st.session_state['document_type'] == "URL":
                    # 주요 세션 정보 구성
                    if not st.session_state['document_source']:
                        st.error("[ERROR] URL 정보가 없습니다.")
                        st.stop()

                    if not is_valid_url(st.session_state['document_source'][0]):
                        st.error("[ERROR] 비정상 URL 입니다.")
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
                    st.session_state['document_source'] = google_search(st.session_state['google_search_query'], num_results=st.session_state['google_search_result_count'], lang=st.session_state['google_search_result_lang'])
                    if not st.session_state['document_source']:
                        st.error("[ERROR] 검색 결과가 없습니다.")
                        st.stop()
                    for url in st.session_state['document_source']:
                        print(f"Loaded URL ------------------------------------------> {url}")
                        loader = WebBaseLoader(
                            web_paths=(url,),
                        )
                        docs_contents.append(loader.load())

                # 문서 분할
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state['chunk_size'], chunk_overlap=st.session_state['chunk_overlap'])
                splits = []
                if st.session_state['document_type'] == "Google Search":
                    for doc_contents in docs_contents:
                        splits.append(text_splitter.split_documents(doc_contents))
                else:
                    splits = text_splitter.split_documents(docs_contents)

                # 모든 청크 벡터화 및 메타데이터 준비 (for Pinecone에 임베딩)
                documents_and_metadata = []
                if st.session_state['document_type'] == "Google Search":
                    for i, docs in enumerate(splits):
                        for j, doc in enumerate(docs):
                            document = Document(
                                page_content=doc.page_content,
                                metadata={"id": j + 1, "source": st.session_state['document_source'][i]}
                            )
                            documents_and_metadata.append(document)
                else:
                    for i, doc in enumerate(splits):
                        document = Document(
                            page_content=doc.page_content,
                            metadata={"id": i + 1, "source": st.session_state['document_source']}
                        )
                        documents_and_metadata.append(document)

                ### Debugging Print
                print(f"documents_and_metadata 개수 ---------------> {len(documents_and_metadata)}")
                
                st.write(f"Documents Chunks: {len(documents_and_metadata)}")

                # Pinecone 관련 설정
                pc = Pinecone()
                pinecone_index_name = 'perplexis'

                # Pinecone Index 초기화 (삭제)
                if pinecone_index_name in pc.list_indexes().names():
                    pc.delete_index(pinecone_index_name)

                # Pinecone Index 생성
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
                    documents_and_metadata,
                    st.session_state['embeddings'],
                    index_name=pinecone_index_name
                )
                
                # 주어진 문서 내용 처리(임베딩)
                if st.session_state['rag_search_type'] == "similarity_score_threshold":
                    search_kwargs = {"k": int(st.session_state['rag_top_k']), "score_threshold": float(st.session_state['rag_score'])}
                elif st.session_state['rag_search_type'] == "similarity":
                    search_kwargs = {"k": int(st.session_state['rag_top_k'])}
                elif st.session_state['rag_search_type'] == "mmr":
                    search_kwargs = {"k": int(st.session_state['rag_top_k']), "fetch_k": 20, "lambda_mult": 0.5}
                st.session_state['retriever'] = vectorstore.as_retriever(search_type=st.session_state['rag_search_type'], search_kwargs=search_kwargs)
                
                if st.session_state['retriever']:
                    st.success("Embedding 완료!")
                else:
                    st.error("[ERROR] Embedding 실패!")
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

#-----------------------------------------------------------------------------------------------------------

    # 메인 창 로딩 가능 여부(retriever 객체 존재) 확인
    try:
        if not (st.session_state['retriever'] and st.session_state['session_id']) or not st.session_state['is_analyzed']:
            st.stop()
    except Exception as e:
        print(f"[Exception]: {e}")
        st.markdown("좌측 사이드바에서 필수 정보를 입력하세요.")
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

            st.session_state['chat_history_user'].append(user_input)
            st.session_state['chat_history_ai'].append(ai_response)
            st.session_state['chat_history_llm_model_name'].append(llm_model_name)
            st.session_state['chat_history_rag_contexts'].append(rag_contexts)
            st.session_state['chat_history_temperature'].append(st.session_state['temperature'])

    ### container_history 처리
    if st.session_state['chat_history_ai']:
        with container_history:
            for i in range(len(st.session_state['chat_history_ai'])):
                message(st.session_state["chat_history_user"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["chat_history_ai"][i], key=str(i))
                
                llm_model_name = st.session_state['chat_history_llm_model_name'][i]
                temperature = st.session_state['chat_history_temperature'][i]
                
                st.write(f"Embeddings: {st.session_state.get('selected_embeddings', 'Unknown Embeddings')} / AI: {st.session_state.get('selected_ai', 'Unknown AI')} / LLM: {llm_model_name} / Temperature: {temperature} / RAG Contexts: {len(st.session_state['chat_history_rag_contexts'][-1])}")
                st.write(f"RAG Search Type: {st.session_state.get('rag_search_type', 'Unknown')} / RAG Score: {st.session_state.get('rag_score', "Unknown")} / RAG TOP-K: {st.session_state.get('rag_top_k', "Unknown")} / Pinecone Metric: {st.session_state.get('pinecone_metric', 'Unknown')}")
                
                # 소스 데이터 표시
                if st.session_state["chat_history_rag_contexts"][i]:
                    with st.expander("Show Contexts"):
                        for context in st.session_state["chat_history_rag_contexts"][i]:
                            st.write(context)

#----------------------------------------------------

if __name__ == "__main__":
    # Error 메세지/코드 숨기기
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
