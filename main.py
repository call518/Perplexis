from modules.nav import Navbar
import streamlit as st

st.set_page_config(page_title="Perplexis:Home", page_icon=":books:", layout="wide")
st.title(f'🔥 _:red[Perplexis]_')

def main():
    Navbar()

    st.markdown("""
    # RAG & Chat Application Guide
    
    ## Chat 지원 기능
    
    ### AI 역할 선택
    - 60여가지 AI 역할 선택 (README 참조)
        
    ### AI 서비스 제공자 선택
    - Ollama
        - Ollama API URL 입력
    - OpenAI
        - OpenAI API URL 입력
        - OpenAI API Key 입력
        
    ### LLM 옵션 조정
    
    #### Ollama
    - LLM 모델 선택
        - gemma2:2b
        - gemma2:9b
        - gemma2:27b
        - mistral:7b
        - llama3.2:1b
        - llama3.2:3b
        - codegemma:2b
        - codegemma:7b
    - Temperature 조정
    - top_p 조정
    - repeat_penalty 조정
    - num_ctx 조정
    - num_predict 조정
    #### OpenAI
    - LLM 모델 선택
        - gpt-4o-mini
        - gpt-4o
        - gpt-4-turbo
        - gpt-4
        - gpt-3.5-turbo
    - Temperature 조정
    - top_p 조정
    - frequency_penalty 조정
    - max_tokens 조정
    - presence_penalty 조정
            
    ## RAG 지원 기능
    
    ### Chat 지원 기능 포함

    ### RAG 분석 데이터 종류 선택    
    -  URL
        - 분석 대상 URL 입력 폼
    
    - File Upload
        - 분석 대상 로컬 파일(TXT, PDF 지원) 업로드
    
    - Google Search
        - 검색 키워드 입력 폼
        - 사용자 커스텀 문서 URL 입력
        - 검색 결과 페이지 수 입력
        - 검색 문서 언어 선택
    
    ### Embedding 서비스 제공자 선택
    - Ollama
    - OpenAI
    
    ### VectorDB
    
    - 저장소 선택
        - PGVector
        - ChromaDB
        - Pinecone
    
    - 문서 조각
        - Chunk Size 조정
        - Chunk Overlap 조정
    
    - VectorDB 초기화 여부 선택
        - `Reset {VectorDB종료}` 체크 박스
        
    - 유사도(Similarity) 알고리즘 선택
        - PGVector
            - cosine
            - euclidean
            - dotproduct
        - ChromaDB
            - cosine
            - l2
            - ip
        - Pinecone
            - cosine
            - l2
            - inner

    - 유사도(Similarity) 평가 방법 선택
        - similarity
        - similarity_score_threshold
        - mmr

    - 기타 RAG 옵션
        - top_k
        - score_threshold

    ### Embedding 모델 선택
    - Ollama
        - bge-m3:56m
        - all-minilm:22m
        - all-minilm:33m
        - nomic-embed-text
        - mxbai-embed-large
        - gemma2:2b
        - gemma2:9b
        - gemma2:27b
        - llama3:8b
        - gemma3.2:1b
        - gemma3.2:3b
    - OpenAI
        - text-embedding-3-small
        - text-embedding-3-large
        - text-embedding-ada-002

    """)


if __name__ == '__main__':
    main()