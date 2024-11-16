# Perplexis

`Perplexis` is Private [Perplexity](https://www.perplexity.ai/)

# Configure

```bash
$ cp -a .streamlit.template .streamlit
$ vi .streamlit/secrets.toml

# Update each API key with your own value
[KEYS]
OPENAI_API_KEY = "{your openai api key}"
PINECONE_API_KEY = "{your pinecone api key}"
LANGCHAIN_API_KEY = "{your langchain api key}"
```
