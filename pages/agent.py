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
from streamlit.runtime.scriptrunner import get_script_run_ctx

import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage

import os
import requests
import time

from langchain.callbacks.base import BaseCallbackHandler  # 추가

# from langchain.agents.agent_toolkits import create_spark_sql_agent

### Agent 관련 모듈 추가 임포트
from langchain.agents import initialize_agent, load_tools, Tool, AgentType
from langchain.agents.format_scratchpad import format_log_to_messages

from langchain.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
# from langchain.chains.llm_math.base import LLMMathChain

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
        "SERPAPI_API_KEY": {"secret_key": "SERPAPI_API_KEY", "default_value": ""},
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
    'ai_agent_chain': None,
    'ai_agent_type_name': None,
    'chat_memory': None,
    # 'chat_memory_placeholder': None,
    'ui_chat_history_array': [],
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

#-------------------- Agetn 관련 함수 및 Tool 정의 --------------------
### Agent 객체 생성 함수
def create_agent_chain():
    if st.session_state['llm'] is None:
        st.error("Please set the AI LLM model.")
        st.stop()
        
    ### Define AI Agent and TOols
    if st.session_state.get('chat_memory', None) is None:
        st.session_state['chat_memory'] = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)

    # if st.session_state.get('chat_memory_placeholder', None) is None:
    #     st.session_state['chat_memory_placeholder'] = MessagesPlaceholder(variable_name="chat_memory")

    # common_system_prompt = "Write final answer in Korean."
    common_system_prompt = "최종 답변은 반드시 한국어로 작성해야 한다."
    # common_system_prompt = "Write final answer in Korean with as much detail as possible."
    # st.session_state['system_prompt_content'] = get_ai_role_and_sysetm_prompt(st.session_state.get('ai_role', "General AI Assistant")) + " Write final answer in Korean."
    st.session_state['system_prompt_content'] = get_ai_role_and_sysetm_prompt(st.session_state.get('ai_role', "General AI Assistant")) + " " + common_system_prompt
    print(f"[DEBUG] (system_prompt_content) -------> {st.session_state['system_prompt_content']}")
    system_message = SystemMessage(content=st.session_state['system_prompt_content'])

    PREFIX = '''You are an AI data scientist. You have done years of research in studying all the AI algorthims. You also love to write. On free time you write blog post for articulating what you have learned about different AI algorithms. Do not forget to include information on the algorithm's benefits, disadvantages, and applications. Additionally, the blog post should explain how the algorithm advances model reasoning by a whopping 70% and how it is a plug in and play version, connecting seamlessly to other components.
    '''

    FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:
    '''
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    '''

    When you have gathered all the information regarding AI algorithm, just write it to the user in the form of a blog post.

    '''
    Thought: Do I need to use a tool? No
    AI: [write a blog post]
    '''
    """

    SUFFIX = '''

    Begin!

    Previous conversation history:
    {chat_history}

    Instructions: {input}
    {agent_scratchpad}
    '''
    
    ### Agent Tool 설정
    # tool_search = DuckDuckGoSearchRun()
    tool_search = SerpAPIWrapper()
    tool_arxiv = ArxivAPIWrapper()
    tool_wikipedia = WikipediaAPIWrapper()
    # tool_calculator = LLMMathChain.from_llm(llm=st.session_state['llm'], verbose=True)
    tool_wa = WA(app_id=os.environ["WA_API_KEY"])

    agent_tools = [
        Tool(
            name = "Web Search",
            func = tool_search.run,
            # description = f"A versatile tool for searching the Internet, useful for finding information on world events, technical documentation, software installation guides, and troubleshooting technical issues. Best used with precise queries.",
            description = f"useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name = "Arxiv",
            func = tool_arxiv.run,
            description = "useful when you need an answer about encyclopedic general knowledge"
        ),
        Tool(
            name = "Wikipedia",
            func = tool_wikipedia.run,
            description = "useful when you need an answer about encyclopedic general knowledge"
        ),
        # Tool(
        #     name = "Calculator",
        #     func = tool_calculator.run,
        #     description = "useful for when you need to answer questions about math"
        # ),
        Tool(
            name="Wolfram|Alpha API",
            func=tool_wa.run,
            description="Wolfram|Alpha API. It's super powerful Math tool. Use it for simple & complex math tasks."
        ),
    ]

    # # SELF_ASK_WITH_SEARCH 에이전트는 단 하나의 도구만 허용하므로 Web Search만 사용
    # if st.session_state['ai_agent_type_name'] == "SELF_ASK_WITH_SEARCH":
    #     agent_tools = [
    #         Tool(
    #             name = "Intermediate Answer",
    #             func = tool_search.run,
    #             description = "useful for when you need to answer questions about current events. You should ask targeted questions",
    #         )
    #     ]

    ### Agent 객체 생성
    agent_chain = initialize_agent(
        agent = getattr(AgentType, st.session_state['ai_agent_type_name']),
        # agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, # <---- 사용금지 (동작 문제는 없으나, chat memory 관련 코드 전면 수정 필요)
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # agent=AgentType.SELF_ASK_WITH_SEARCH,
        # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        # agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        # agent=AgentType.REACT_DOCSTORE,
        # agent=AgentType.OPENAI_FUNCTIONS,
        # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        tools=agent_tools,
        llm=st.session_state['llm'],
        early_stopping_method='generate',
        agent_kwargs={
            # "system_message": system_message,
            "prefix": st.session_state['system_prompt_content'],
            # "format_instructions": FORMAT_INSTRUCTIONS,
            # "suffix": SUFFIX
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_memory")],
            # "memory_prompts": [st.session_state['chat_memory_placeholder']],
        },
        handle_parsing_errors=True,
        agent_max_iterations=st.session_state['agent_max_iterations'],
        memory=st.session_state['chat_memory'],
        verbose=True, # I will use verbose=True to check process of choosing tool by Agent
    )
    
    return agent_chain

#--------------------------------------------------

def main():
    """
    메인 함수로서, 스트림릿 인터페이스를 구성하고 
    AI 모델 파라미터 설정 및 채팅 기능을 제공합니다.
    """
    # 사이드바 영역 구성
    with st.sidebar:
        st.title("Parameters")

        st.session_state['selected_ai'] = st.radio("**:blue[AI]**", ("Ollama", "OpenAI"), index=0, disabled=st.session_state['is_analyzed'])

        if st.session_state.get('selected_ai', "Ollama") == "OpenAI":
            os.environ["OPENAI_API_KEY"] = st.text_input("**:red[OpenAI API Key]** [Learn more](https://platform.openai.com/docs/quickstart)", value=os.environ["OPENAI_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
            os.environ["OPENAI_BASE_URL"] = st.text_input("OpenAI API URL", value=os.environ["OPENAI_BASE_URL"], disabled=st.session_state['is_analyzed'])

        st.session_state['ai_role'] = st.selectbox("Role of AI", get_ai_role_and_sysetm_prompt(only_key=True), index=0, disabled=st.session_state['is_analyzed'])

        ai_agent_types = [
            "ZERO_SHOT_REACT_DESCRIPTION",
            "STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION",
            # "CHAT_ZERO_SHOT_REACT_DESCRIPTION",
            #"CHAT_CONVERSATIONAL_REACT_DESCRIPTION",
            #"CONVERSATIONAL_REACT_DESCRIPTION",
            #"SELF_ASK_WITH_SEARCH",
            #"REACT_DOCSTORE",
            #"OPENAI_FUNCTIONS",
            #"OPENAI_MULTI_FUNCTIONS",
        ]

        st.session_state['ai_agent_type_name'] = st.selectbox("Agent Type", ai_agent_types, index=0, disabled=st.session_state['is_analyzed'])

        os.environ["WA_API_KEY"] = st.text_input("**:red[Wolfram-Alpha API key]** [Link](https://products.wolframalpha.com/api/)", value=os.environ["WA_API_KEY"], type="password", disabled=st.session_state['is_analyzed'])
        
        st.session_state['agent_max_iterations'] = st.number_input("Agent Max Iterations", min_value=1, max_value=10, value=st.session_state.get('agent_max_iterations', 3), step=1, disabled=st.session_state['is_analyzed'])
        
        if st.session_state.get('selected_embedding_provider', "Ollama") == "Ollama" or st.session_state.get('selected_ai', "Ollama") == "Ollama":
            os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama API URL", value=os.environ["OLLAMA_BASE_URL"], disabled=st.session_state['is_analyzed'])

        col_ai_llm, col_ai_temperature = st.sidebar.columns(2)
        with col_ai_llm:
            if st.session_state['selected_ai'] == "OpenAI":
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], index=4, disabled=st.session_state['is_analyzed'])
            else:
                st.session_state['selected_llm'] = st.selectbox("AI LLM", ["gemma2:2b", "gemma2:9b", "gemma2:27b", "call518/gemma2-uncensored-8192ctx:9b", "mistral:7b", "llama3.2:1b", "llama3:8b", "llama3.2:3b", "codegemma:2b", "codegemma:7b", "mistral:7b", "call518/deepseek-r1-32768ctx:8b", "call518/deepseek-r1-32768ctx:14b", "call518/EEVE-8192ctx:10.8b-q4", "call518/EEVE-8192ctx:10.8b-q5", "llama3-groq-tool-use:8b"], index=0, disabled=st.session_state['is_analyzed'])
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

        if st.button("Reset", type='primary'):
            streamlit_js_eval(js_expressions="parent.window.location.reload()")

#-----------------------------------------------------------------------------------------------------------

    if not os.environ['WA_API_KEY']:
        st.error("Please enter the Wolfram|Alpha API Key.")
        st.stop()

    ## Container 선언 순서가 화면에 보여지는 순서 결정
    container_history = st.container()
    container_user_textbox = st.container()

    ### container_user_textbox 처리
    with container_user_textbox:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send', help="Click to send your message", icon=":material/send:", type='primary')

        if submit_button and user_input:
            ### 파라메터 잠금 처리
            st.session_state['is_analyzed'] = True
            
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
            
            ### Agent 객체 생성
            agent_chain = st.session_state['ai_agent_chain']
            if agent_chain is None:
                agent_chain = create_agent_chain()

            with st.spinner('Thinking...'):
                start_time = time.perf_counter()
                # response = agent_chain.run(user_input)
                response = agent_chain.run(input=user_input)
                response = str(response).strip() # convert to plain string
                # print(f"[DEBUG] (response) ====> {response}")
                st.session_state['chat_response'] = response
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

            st.session_state["ui_chat_history_array"].append({
                "role": "user",
                "content": user_input.replace("\n", "\n\n"),
            })
            
            st.session_state["ui_chat_history_array"].append({
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
    if st.session_state.get('ui_chat_history_array', None) is not None:
        with container_history:
            for message in st.session_state['ui_chat_history_array']:
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
    
    # main()
    
    # Error 메세지/코드 숨기기
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
