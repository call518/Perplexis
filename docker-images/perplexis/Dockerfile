FROM rockylinux:9.3

COPY Dockerfile /

RUN dnf install -y epel-release

RUN dnf install -y \
        passwd \
        psmisc \
        wget \
        curl-minimal \
        tcpdump \
        procps-ng \
        tree \
        jq \
        lsof \
        net-tools \
        vim \
        findutils \
        lsof \
        openssh-server \
        chrony \
        rsync \
        rsyslog \
        sudo \
        crontabs \
        cronie \
        at \
        libselinux-utils \
        iputils \
        git \
        telnet \
        postgresql \
		python39 \
		python3.9-pip \
        && dnf clean all

RUN pip3.9 install langchain-community bs4 langchain_pinecone pinecone-client[grpc] langchain-openai langchain_ollama streamlit-chat streamlit-js-eval googlesearch-python chromadb pysqlite3-binary pypdf pymupdf rapidocr-onnxruntime langchain-experimental langchain_postgres psycopg_binary sqlalchemy

COPY bashrc /tmp/bashrc

RUN cat /tmp/bashrc >> /etc/bashrc && rm /tmp/bashrc

COPY init.sh /

ENTRYPOINT [ "/init.sh" ]