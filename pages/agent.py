from modules.nav import Navbar
from modules.common_functions import get_ai_role_and_sysetm_prompt
from modules.common_functions import get_country_name_by_code
from modules.common_functions import get_max_value_of_model_max_tokens
from modules.common_functions import get_max_value_of_model_num_ctx
from modules.common_functions import get_max_value_of_model_num_predict
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
from langchain.chains import ConversationalRetrievalChain

from langchain_ollama import OllamaLLM

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

import os
import requests
import time

from langchain.callbacks.base import BaseCallbackHandler  # 추가

### Agent 관련 모듈 추가 임포트
from langchain.agents import initialize_agent, load_tools, Tool
from langchain.tools import DuckDuckGoSearchRun

#--------------------------------------------------

# 페이지 정보 정의
st.set_page_config(page_title="Perplexis:Agent", page_icon=":books:", layout="wide")
st.title(":material/support_agent: _:red[Perplexis]_ Agent")

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
        "WA_API_KEY": {"secret_key": "WA_API_KEY", "default_value": ""},
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
    # 'chat_memory': None,
    'agent_chat_history': [],
    # 'llm_chain_conversation': None,
    'chat_response': None,
    'session_id': str(uuid.uuid4()),
    'is_analyzed': False,
    'agent_max_iterations': 3,
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
    'temperature': 0.20,
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

### Wolfram|Alpha API
class WA:
    def __init__(self, app_id):
        self.url = f"http://api.wolframalpha.com/v1/result?appid={app_id}&i="
    def run(self, query):
        query = query.replace("+", " plus ").replace("-", " minus ") # '+' and '-' are used in URI and cannot be used in request
        result = requests.post(f"{self.url}{query}")
        if not result.ok:
            raise Exception("Cannot call WA API.")
        return result.text

#--------------------------------------------------

def main():
    """
    메인 함수로서, 스트림릿 인터페이스를 구성하고 
    AI 모델 파라미터 설정 및 채팅 기능을 제공합니다.
    """
    # 사이드바 영역 구성
    with st.sidebar:
        st.title("Parameters")

        st.session_state['ai_role'] = st.selectbox("Role of AI", get_ai_role_and_sysetm_prompt(only_key=True), index=0)

        st.session_state['selected_ai'] = st.radio("**:blue[AI]**", ("Ollama", "OpenAI"), index=0, disabled=st.session_state['is_analyzed'])

        if st.session_state.get('selected_ai', "Ollama") == "OpenAI":
            os.environ["OPENAI_API_KEY"] = st.text_input("**:red[OpenAI API Key]** [Learn more](https://platform.openai.com/docs/quickstart)", value=os.environ["OPENAI_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
            os.environ["OPENAI_BASE_URL"] = st.text_input("OpenAI API URL", value=os.environ["OPENAI_BASE_URL"], disabled=st.session_state['is_analyzed'])

        os.environ["WA_API_KEY"] = st.text_input("**:red[Wolfram-Alpha API key]** [Link](https://products.wolframalpha.com/api/)", value=os.environ["WA_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
        
        st.session_state['agent_max_iterations'] = st.number_input("agent_max_iterations", min_value=1, max_value=10, value=st.session_state.get('agent_max_iterations', 3), step=1, disabled=st.session_state['is_analyzed'])
        
        if st.session_state.get('selected_embedding_provider', "Ollama") == "Ollama" or st.session_state.get('selected_ai', "Ollama") == "Ollama":
            os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama API URL", value=os.environ["OLLAMA_BASE_URL"], disabled=st.session_state['is_analyzed'])

        col_ai_llm, col_ai_temperature = st.sidebar.columns(2)
        with col_ai_llm:
            if st.session_state['selected_ai'] == "OpenAI":
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], index=4, disabled=st.session_state['is_analyzed'])
            else:
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gemma2:2b", "gemma2:9b", "gemma2:27b", "call518/gemma2-uncensored-8192ctx:9b", "mistral:7b", "llama3.2:1b", "llama3:8b", "llama3.2:3b", "codegemma:2b", "codegemma:7b", "mistral:7b", "call518/deepseek-r1-32768ctx:8b", "call518/deepseek-r1-32768ctx:14b", "call518/EEVE-8192ctx:10.8b-q4", "call518/EEVE-8192ctx:10.8b-q5"], index=0, disabled=st.session_state['is_analyzed'])
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
                st.session_state['llm_ollama_num_ctx'] = st.number_input("num_ctx", min_value=256, max_value=get_max_value_of_model_num_ctx(st.session_state['selected_llm']), value=int(get_max_value_of_model_num_ctx(st.session_state['selected_llm']) / 2), step=512, disabled=st.session_state['is_analyzed'])
                print(f"[DEBUG] (llm_ollama_num_ctx) {st.session_state['llm_ollama_num_ctx']}")
            with col_llm_ollama_num_predict:
                st.session_state['llm_ollama_num_predict'] = st.number_input("num_predict", value=int(get_max_value_of_model_num_predict(st.session_state['selected_llm']) / 2), disabled=st.session_state['is_analyzed'])
                print(f"[DEBUG] (llm_ollama_num_predict) {st.session_state['llm_ollama_num_predict']}")

        ### Set OpenAI API Key
        if st.session_state.get('selected_ai', "Ollama") == "OpenAI":
            if not os.environ["OPENAI_API_KEY"]:
                st.error("Please enter the OpenAI API Key.")
                st.stop()

        # # AI 모델 선택 및 초기화
        # if st.session_state['selected_ai'] == "OpenAI":
        #     st.session_state['llm'] = ChatOpenAI(
        #         base_url = os.environ["OPENAI_BASE_URL"],
        #         model = st.session_state.get('selected_llm', "gpt-3.5-turbo"),
        #         temperature = st.session_state.get('temperature', 0.00),
        #         cache = False,
        #         streaming = False,
        #         presence_penalty = st.session_state.get('llm_openai_presence_penalty', 1.00),
        #         frequency_penalty = st.session_state.get('llm_openai_frequency_penalty', 1.00),
        #         stream_usage = False,
        #         n = 1,
        #         top_p = st.session_state.get('llm_top_p', 0.50),
        #         max_tokens = st.session_state.get('llm_openai_max_tokens', 1024),
        #         verbose = True,
        #     )
        # else:
        #     st.session_state['llm'] = OllamaLLM(
        #         base_url = os.environ["OLLAMA_BASE_URL"],
        #         model = st.session_state.get('selected_llm', "gemma2:9b"),
        #         temperature = st.session_state.get('temperature', 0.00),
        #         cache = False,
        #         num_ctx = st.session_state.get('llm_ollama_num_ctx', 1024),
        #         num_predict = st.session_state.get('llm_ollama_num_predict', -1),
        #         # num_gpu = None,
        #         # num_thread = None,
        #         # repeat_last_n = None,
        #         repeat_penalty = st.session_state.get('llm_ollama_repeat_penalty', 1.00),
        #         # tfs_z = None,
        #         # top_k = None,
        #         top_p = st.session_state.get('llm_top_p', 0.80),
        #         # format = "", # Literal['', 'json'] (default: "")
        #         verbose = True,
        #     )

        if st.button("Reset", type='primary'):
            streamlit_js_eval(js_expressions="parent.window.location.reload()")

#-----------------------------------------------------------------------------------------------------------

    ### Web Search Tool 객체 생성
    search = DuckDuckGoSearchRun()
    # Web Search Tool
    search_tool = Tool(
        name = "Web Search",
        func=search.run,
        # description = f"A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions.",
        # description = f"A versatile tool for searching the Internet, useful for finding information on world events, technical documentation, software installation guides, and troubleshooting technical issues. Best used with precise queries.",
        # description = f"{st.session_state['ai_role']} A versatile tool for searching the Internet, useful for finding information on world events, technical documentation, software installation guides, and troubleshooting technical issues. Best used with precise queries. The final response should be provided in Korean whenever possible.",
        description = f"{st.session_state['ai_role']} A versatile tool for searching the Internet, useful for finding information on world events, technical documentation, software installation guides, and troubleshooting technical issues. Best used with precise queries. Additionally, responses should be as detailed and comprehensive as possible, providing in-depth explanations, step-by-step instructions, and relevant examples where applicable. The final response should be provided in Korean whenever possible.",
    )

    ### Wolfram|Alpha API Tool 객체 생성
    wa = WA(app_id=os.environ["WA_API_KEY"])
    wa_tool = Tool(
        name="Wolfram|Alpha API",
        func=wa.run,
        description="Wolfram|Alpha API. It's super powerful Math tool. Use it for simple & complex math tasks."
    )

    # system_prompt_content = "You are a chatbot having a conversation with a human."
    # system_prompt_content = "I want you to act as an academician. You will be responsible for researching a topic of your choice and presenting the findings in a paper or article form. Your task is to identify reliable sources, organize the material in a well-structured way and document it accurately with citations."

    st.session_state['system_prompt_content'] = get_ai_role_and_sysetm_prompt(st.session_state.get('ai_role', "General AI Assistant"))
    system_prompt = st.session_state['system_prompt_content']

    # ### Chat Memory 객체 생성 (최초 1회만 생성)
    # if st.session_state.get('chat_memory', None) is None:
    #     st.session_state['chat_memory'] = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)

    # ### Chat Prompt 객체 생성
    # prompt = ChatPromptTemplate(
    #     messages=[
    #         SystemMessagePromptTemplate.from_template(system_prompt),
    #         MessagesPlaceholder(variable_name="chat_memory"),
    #         HumanMessagePromptTemplate.from_template("{question}")
    #     ]
    # )

    ## Container 선언 순서가 화면에 보여지는 순서 결정
    container_history = st.container()
    container_user_textbox = st.container()

    ### container_user_textbox 처리
    with container_user_textbox:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send', help="Click to send your message", icon=":material/send:", type='primary')

        if submit_button and user_input:
            # streaming_placeholder = st.empty()  # 토큰 실시간 출력을 위한 컨테이너
            # callback_handler = StreamlitCallbackHandler(placeholder=streaming_placeholder)

            # LLM 재생성: streaming=True 및 callbacks 적용
            if st.session_state['selected_ai'] == "OpenAI":
                st.session_state['llm'] = ChatOpenAI(
                    base_url = os.environ["OPENAI_BASE_URL"],
                    model = st.session_state.get('selected_llm', "gpt-3.5-turbo"),
                    temperature = st.session_state.get('temperature', 0.00),
                    cache = False,
                    streaming = False,
                    # callbacks = [callback_handler],
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
                    streaming = False,
                    # callbacks = [callback_handler],
                    num_ctx = st.session_state.get('llm_ollama_num_ctx', 1024),
                    num_predict = st.session_state.get('llm_ollama_num_predict', -1),
                    repeat_penalty = st.session_state.get('llm_ollama_repeat_penalty', 1.00),
                    top_p = st.session_state.get('llm_top_p', 0.80),
                    verbose = True,
                )

            # ### Chat Conversation 객체 생성        
            # st.session_state['llm_chain_conversation'] = LLMChain(
            #     llm=st.session_state['llm'],
            #     prompt=prompt,
            #     memory=st.session_state['chat_memory'],
            #     verbose=True,
            # )

            # # llm_chain_conversation 재생성 (새 llm 적용)
            # st.session_state['llm_chain_conversation'] = LLMChain(
            #     llm=st.session_state['llm'],
            #     prompt=ChatPromptTemplate(
            #         messages=[
            #             SystemMessagePromptTemplate.from_template(st.session_state['system_prompt_content']),
            #             MessagesPlaceholder(variable_name="chat_memory"),
            #             HumanMessagePromptTemplate.from_template("{question}")
            #         ]
            #     ),
            #     memory=st.session_state['chat_memory'],
            #     verbose=True,
            # )

            ### Agent 객체 생성
            agent = initialize_agent(
                agent="zero-shot-react-description",
                # agent="structured-chat-zero-shot-react-description",
                tools=[wa_tool, search_tool],
                llm=st.session_state['llm'],
                # llm=st.session_state['llm_chain_conversation'],
                verbose=True, # I will use verbose=True to check process of choosing tool by Agent
                agent_max_iterations=3
            )

            with st.spinner('Thinking...'):
                start_time = time.perf_counter()
                # response = st.session_state['llm_chain_conversation'].run({"question": user_input})
                response = agent.run(user_input)
                response = str(response)  # convert to plain string
                print(f"[DEBUG] (response) ==> {response}")
                st.session_state['chat_response'] = response
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

            # streaming_placeholder.empty()  # 추가: 스트리밍 출력 후 placeholder 비우기
            
            st.session_state["agent_chat_history"].append({
                "role": "user",
                "content": user_input.replace("\n", "\n\n"),
            })
            
            st.session_state["agent_chat_history"].append({
                "role": "assistant",
                "content": st.session_state['chat_response'],
                "agent_max_iterations": st.session_state["agent_max_iterations"],
                "ai_role": st.session_state["ai_role"],
                "selected_ai": st.session_state["selected_ai"],
                "selected_llm": st.session_state["selected_llm"],
                "temperature": st.session_state["temperature"],
                "llm_top_p": st.session_state["llm_top_p"],
                "llm_openai_presence_penalty": st.session_state["llm_openai_presence_penalty"],
                "llm_openai_frequency_penalty": st.session_state["llm_openai_frequency_penalty"],
                "llm_openai_max_tokens": st.session_state["llm_openai_max_tokens"],
                "llm_ollama_repeat_penalty": st.session_state["llm_ollama_repeat_penalty"],
                "llm_ollama_num_ctx": st.session_state["llm_ollama_num_ctx"],
                "llm_ollama_num_predict": st.session_state["llm_ollama_num_predict"],
                "elapsed_time": elapsed_time,
            })

    ### container_history 처리
    if st.session_state.get('agent_chat_history', None) is not None:
        with container_history:
            for message in st.session_state['agent_chat_history']:
                if message['role'] == "user":
                    with st.chat_message("user"):
                        st.markdown("**<span style='color: blue;'>You</span>**", unsafe_allow_html=True)
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(f"**<span style='color: green;'>{message['ai_role']}</span>**", unsafe_allow_html=True)
                        st.write(message['content'])
                        with st.expander("Show Metadata"):
                            st.markdown(
                                f"""
                                <div style="color: #2E9AFE; border: 1px solid #ddd; padding: 5px; background-color: #f9f9f9; border-radius: 5px; width: 100%;">
                                    - <b>AI</b>: <font color=red>{message['selected_ai']}</font><br>
                                    - <b>Agent Args(Common) agent_max_iterations</b>: <font color=black>{message['agent_max_iterations']}</font><br>
                                    - <b>LLM Model(Common)</b>: <font color=black>{message['selected_llm']}</font><br>
                                    - <b>LLM Args(Common) temperature</b>: <font color=black>{message['temperature']}</font><br>
                                    - <b>LLM Args(Common) top_p</b>: <font color=black>{message['llm_top_p']}</font><br>
                                    - <b>LLM Args(OpenAI) presence_penalty</b>: <font color=black>{message['llm_openai_presence_penalty']}</font><br>
                                    - <b>LLM Args(OpenAI) frequency_penalty</b>: <font color=black>{message['llm_openai_frequency_penalty']}</font><br>
                                    - <b>LLM Args(OpenAI) max_tokens</b>: <font color=black>{message['llm_openai_max_tokens']}</font><br>
                                    - <b>LLM Args(Ollama) repeat_penalty</b>: <font color=black>{message['llm_ollama_repeat_penalty']}</font><br>
                                    - <b>LLM Args(Ollama) num_ctx</b>: <font color=black>{message['llm_ollama_num_ctx']}</font><br>
                                    - <b>LLM Args(Ollama) num_predict</b>: <font color=black>{message['llm_ollama_num_predict']}</font><br>
                                    - <b>Elapsed time</b>: <font color=black>{message['elapsed_time']:.0f} sec</font><br>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.write("")

#----------------------------------------------------

if __name__ == "__main__":
    Navbar()
    
    main()
    # # Error 메세지/코드 숨기기
    # try:
    #     main()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
