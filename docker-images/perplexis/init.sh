#!/bin/bash

set -e

if [ -z $PERPLEXIS_BRANCH ]; then
    PERPLEXIS_BRANCH="main"
fi
git clone https://github.com/call518/Perplexis.git /Perplexis
cd /Perplexis && git checkout $PERPLEXIS_BRANCH && rm -rf .git

if [ $# -eq 0 ]; then
    SECRETS_TOML="/Perplexis/.streamlit/secrets.toml"
    cp -a /Perplexis/.streamlit.template/secrets.toml $SECRETS_TOML

    if [ -n "$OLLAMA_BASE_URL" ]; then
        sed -i "s|^OLLAMA_BASE_URL = .*|OLLAMA_BASE_URL = \"$OLLAMA_BASE_URL\"|g" $SECRETS_TOML
    fi

    if [ -n "$OPENAI_BASE_URL" ]; then
        sed -i "s|^OPENAI_BASE_URL = .*|OPENAI_BASE_URL = \"$OPENAI_BASE_URL\"|g" $SECRETS_TOML
    fi

    if [ -n "$OPENAI_API_KEY" ]; then
        sed -i "s|^OPENAI_API_KEY = .*|OPENAI_API_KEY = \"$OPENAI_API_KEY\"|g" $SECRETS_TOML
    fi

    if [ -n "$PINECONE_API_KEY" ]; then
        sed -i "s|^PINECONE_API_KEY = .*|PINECONE_API_KEY = \"$PINECONE_API_KEY\"|g" $SECRETS_TOML
    fi

    if [ -n "$LANGCHAIN_API_KEY" ]; then
        sed -i "s|^LANGCHAIN_API_KEY = .*|LANGCHAIN_API_KEY = \"$LANGCHAIN_API_KEY\"|g" $SECRETS_TOML
    fi

    cd /Perplexis
    
    streamlit run main.py --server.port 8501
else
    exec "$@"
fi