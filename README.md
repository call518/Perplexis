# Perplexis

`Perplexis` is Private AI Search Engine (like Perplexity)

## Chat & RAG Application Guide

Welcome to the Chat & RAG Application. This application allows you to interact with a chatbot equipped with RAG (Retrieval-Augmented Generation) capabilities. Below is a guide to its functions, features, and how to use it.

### Features

- **Chat Mode**: Engage in general conversation with an AI assistant.
- **RAG Mode**: Provides answers to questions by retrieving information from selected documents or web sources.
- **Document Upload**: Upload text or PDF documents to use as context in RAG mode.
- **Google Search Integration**: Enhances answers to questions by searching for information on the web.
- **Embedding and AI Model Selection**: Choose from various embedding and AI models for text processing and generation.

### How to Use

1. **Select Mode**: Choose 'Chat' or 'RAG' mode from the sidebar.
2. **Set Parameters**: Adjust preferred settings such as AI model, embedding model, temperature, etc.
3. **Upload Documents (RAG Mode)**: Upload documents or enter a URL to use as context in 'RAG' mode.
4. **Enter Questions**: Type questions into the input field at the bottom of the chat interface.
5. **Review Responses**: The AI assistant will respond based on the mode and the provided context.

### Notes

- Required API keys must be set up as needed.
- The application supports multiple languages selectable from the sidebar.
- In 'RAG' mode, context from documents or search results is used to provide more accurate answers.

Enjoy your time with the Chat & RAG Application!

# Tested Environments

- Python : 3.12.7
- ollama version is 0.4.1

# Configure & Run

## Setup Default API Keys (Optional)

You can be entered directly in the Web-UI, or pre-configured as shown below.

```bash
$ cp -a .streamlit.template .streamlit

$ vi .streamlit/secrets.toml
### Update each API keys and BASE URLs with your own value
### LANGCHAIN_API_KEY is not mandatory 

[KEYS]
### (Mandatory)
OLLAMA_BASE_URL = "{your ollama api server base url}"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_API_KEY = "{your openai api key}"
PINECONE_API_KEY = "{your pinecone api key}"

### (Optional)
LANGCHAIN_API_KEY = "{your langchain api key}"
```

## Setup PIP Modules

```bash
$ pip install -r requirements.txt
```

## Run App

```bash
$ streamlit run perplexis.py
```

# Screenshots

## Access Web-UI

- http://localhost:8501/

![Main Page](etc/image-001.png)
![Main Page](etc/image-002.png)
