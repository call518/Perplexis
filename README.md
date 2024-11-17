# Perplexis

`Perplexis` is Private [Perplexity](https://www.perplexity.ai/)

# Tested Environments

- Python : 3.12.7 (/w Miniconda)
- ollama version is 0.4.1

# Configure & Run

## Setup API Keys

```bash
$ cp -a .streamlit.template .streamlit

$ vi .streamlit/secrets.toml
# Update each API key with your own value
[KEYS]
OPENAI_API_KEY = "{your openai api key}"
PINECONE_API_KEY = "{your pinecone api key}"
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