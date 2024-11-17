# Perplexis

`Perplexis` is Private [Perplexity](https://www.perplexity.ai/)

# Tested Environments

- Python : 3.12.7 (/w Miniconda)
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